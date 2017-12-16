/**
 *  Copyright (c) 2015 by Contributors
 */
#include "ps/internal/van.h"
#include <thread>
#include <chrono>
#include "ps/base.h"
#include "ps/sarray.h"
#include "ps/internal/postoffice.h"
#include "ps/internal/customer.h"
#include "./network_utils.h"
#include "./meta.pb.h"
#include "./zmq_van.h"
#include "./resender.h"
namespace ps {

// interval in second between to heartbeast signals. 0 means no heartbeat.
// don't send heartbeast in default. because if the scheduler received a
// heartbeart signal from a node before connected to that node, then it could be
// problem.
static const int kDefaultHeartbeatInterval = 0;

Van* Van::Create(const std::string& type) {
  if (type == "zmq") {
    return new ZMQVan();
  } else {
    LOG(FATAL) << "unsupported van type: " << type;
    return nullptr;
  }
}

void Van::Start() {
  // get scheduler info
  scheduler_.hostname = std::string(CHECK_NOTNULL(Environment::Get()->find("DMLC_PS_ROOT_URI")));
  scheduler_.port     = atoi(CHECK_NOTNULL(Environment::Get()->find("DMLC_PS_ROOT_PORT")));
  scheduler_.role     = Node::SCHEDULER;
  scheduler_.id       = kScheduler;
  is_scheduler_       = Postoffice::Get()->is_scheduler();

  // get my node info
  if (is_scheduler_) {
    my_node_ = scheduler_;
  } else {
    auto role = is_scheduler_ ? Node::SCHEDULER :
                (Postoffice::Get()->is_worker() ? Node::WORKER : Node::SERVER);
    const char* nhost = Environment::Get()->find("DMLC_NODE_HOST");
    std::string ip;
    if (nhost) ip = std::string(nhost);
    if (ip.empty()) {
      const char*  itf = Environment::Get()->find("DMLC_INTERFACE");
      std::string interface;
      if (itf) interface = std::string(itf);
      if (interface.size()) {
        GetIP(interface, &ip);
      } else {
        GetAvailableInterfaceAndIP(&interface, &ip);
      }
      CHECK(!interface.empty()) << "failed to get the interface";
    }
    int port = GetAvailablePort();
    const char* pstr = Environment::Get()->find("PORT");
    if (pstr) port = atoi(pstr);
    CHECK(!ip.empty()) << "failed to get ip";
    CHECK(port) << "failed to get a port";
    my_node_.hostname = ip;
    my_node_.role     = role;
    my_node_.port     = port;
    // cannot determine my id now, the scheduler will assign it later
    // set it explicitly to make re-register within a same process possible
    my_node_.id = Node::kEmpty;
  }

  // bind.
  my_node_.port = Bind(my_node_, is_scheduler_ ? 0 : 40);
  PS_VLOG(1) << "Bind to " << my_node_.DebugString();
  CHECK_NE(my_node_.port, -1) << "bind failed";

  // connect to the scheduler
  Connect(scheduler_);

  // for debug use
  if (Environment::Get()->find("PS_DROP_MSG")) {
    drop_rate_ = atoi(Environment::Get()->find("PS_DROP_MSG"));
  }
  // start receiver
  receiver_thread_ = std::unique_ptr<std::thread>(
      new std::thread(&Van::Receiving, this));

  if (!is_scheduler_) {
    // let the scheduler know myself
    Message msg;
    msg.meta.recver = kScheduler;
    msg.meta.control.cmd = Control::ADD_NODE;
    msg.meta.control.node.push_back(my_node_);
    msg.meta.timestamp = timestamp_++;
    Send(msg);
  }
  // wait until ready
  while (!ready_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // resender
  if (Environment::Get()->find("PS_RESEND") && atoi(Environment::Get()->find("PS_RESEND")) != 0) {
    int timeout = 1000;
    if (Environment::Get()->find("PS_RESEND_TIMEOUT")) {
      timeout = atoi(Environment::Get()->find("PS_RESEND_TIMEOUT"));
    }
    resender_ = new Resender(timeout, 10, this);
  }

  if (!is_scheduler_) {
    // start heartbeat thread
    heartbeat_thread_ = std::unique_ptr<std::thread>(
      new std::thread(&Van::Heartbeat, this));
  }
}

void Van::Stop() {
  // stop threads
  Message exit;
  exit.meta.control.cmd = Control::TERMINATE;
  exit.meta.recver = my_node_.id;
  SendMsg(exit);
  receiver_thread_->join();
  if (!is_scheduler_) heartbeat_thread_->join();
  if (resender_) delete resender_;
}

int Van::Send(const Message& msg) {
  int send_bytes = SendMsg(msg);
  CHECK_NE(send_bytes, -1);
  send_bytes_ += send_bytes;
  if (resender_) resender_->AddOutgoing(msg);
  if (Postoffice::Get()->verbose() >= 2) {
    PS_VLOG(2) << msg.DebugString();
  }
  return send_bytes;
}

void Van::Receiving() {
  const char* heartbeat_timeout_val = Environment::Get()->find("PS_HEARTBEAT_TIMEOUT");
  const int heartbeat_timeout
      = heartbeat_timeout_val ? atoi(heartbeat_timeout_val) : kDefaultHeartbeatInterval;
  Meta nodes;  // for scheduler usage
  while (true) {
    Message msg;
    int recv_bytes = RecvMsg(&msg);

    // For debug, drop received message
    if (ready_ && drop_rate_ > 0) {
      unsigned seed = time(NULL) + my_node_.id;
      if (rand_r(&seed) % 100 < drop_rate_) {
        LOG(WARNING) << "Drop message " << msg.DebugString();
        continue;
      }
    }

    CHECK_NE(recv_bytes, -1);
    recv_bytes_ += recv_bytes;
    if (Postoffice::Get()->verbose() >= 2) {
      PS_VLOG(2) << msg.DebugString();
    }
    // duplicated message
    if (resender_ && resender_->AddIncomming(msg)) continue;

    if (!msg.meta.control.empty()) {
      // do some management
      auto& ctrl = msg.meta.control;
      if (ctrl.cmd == Control::TERMINATE) {
        PS_VLOG(1) << my_node_.ShortDebugString() << " is stopped";
        ready_ = false;
        break;
      } else if (ctrl.cmd == Control::ADD_NODE) {
        size_t num_nodes = Postoffice::Get()->num_servers() +
                           Postoffice::Get()->num_workers();
        auto dead_nodes = Postoffice::Get()->GetDeadNodes(heartbeat_timeout);
        std::unordered_set<int> dead_set(dead_nodes.begin(), dead_nodes.end());
        Meta recovery_nodes;  // store recovery nodes
        recovery_nodes.control.cmd = Control::ADD_NODE;
        // assign an id
        if (msg.meta.sender == Meta::kEmpty) {
          CHECK(is_scheduler_);
          CHECK_EQ(ctrl.node.size(), 1);
          if (nodes.control.node.size() < num_nodes) {
            nodes.control.node.push_back(ctrl.node[0]);
          } else {
            // some node dies and restarts
            CHECK(ready_);
            for (size_t i = 0; i < nodes.control.node.size() - 1; ++i) {
              const auto& node = nodes.control.node[i];
              if (dead_set.find(node.id) != dead_set.end() && node.role == ctrl.node[0].role) {
                auto& recovery_node = ctrl.node[0];
                // assign previous node id
                recovery_node.id = node.id;
                recovery_node.is_recovery = true;
                PS_VLOG(1) << "replace dead node " << node.DebugString()
                           << " by node " << recovery_node.DebugString();
                nodes.control.node[i] = recovery_node;
                recovery_nodes.control.node.push_back(recovery_node);
                break;
              }
            }
          }
        }

        // update my id
        for (size_t i = 0; i < ctrl.node.size(); ++i) {
          const auto& node = ctrl.node[i];
          if (my_node_.hostname == node.hostname &&
              my_node_.port == node.port) {
            my_node_ = node;
            std::string rank = std::to_string(Postoffice::IDtoRank(node.id));
#ifdef _MSC_VER
            _putenv_s("DMLC_RANK", rank.c_str());
#else
            setenv("DMLC_RANK", rank.c_str(), true);
#endif
          }
        }

        if (is_scheduler_) {
          time_t t = time(NULL);
          if (nodes.control.node.size() == num_nodes) {
            // sort the nodes according their ip and port,
            std::sort(nodes.control.node.begin(), nodes.control.node.end(),
                      [](const Node& a, const Node& b) {
                        return (a.hostname.compare(b.hostname) | (a.port < b.port)) > 0;
                      });
            // assign node rank
            for (auto& node : nodes.control.node) {
              CHECK_EQ(node.id, Node::kEmpty);
              int id = node.role == Node::SERVER ?
                       Postoffice::ServerRankToID(num_servers_) :
                       Postoffice::WorkerRankToID(num_workers_);
              PS_VLOG(1) << "assign rank=" << id << " to node " << node.DebugString();
              node.id = id;
              Connect(node);
              if (node.role == Node::SERVER) ++num_servers_;
              if (node.role == Node::WORKER) ++num_workers_;
              Postoffice::Get()->UpdateHeartbeat(node.id, t);
            }
            nodes.control.node.push_back(my_node_);
            nodes.control.cmd = Control::ADD_NODE;
            Message back; back.meta = nodes;
            for (int r : Postoffice::Get()->GetNodeIDs(
                     kWorkerGroup + kServerGroup)) {
              back.meta.recver = r;
              back.meta.timestamp = timestamp_++;
              Send(back);
            }
            PS_VLOG(1) << "the scheduler is connected to "
                    << num_workers_ << " workers and " << num_servers_ << " servers";
            ready_ = true;
          } else if (recovery_nodes.control.node.size() > 0) {
            // send back the recovery node
            CHECK_EQ(recovery_nodes.control.node.size(), 1);
            Connect(recovery_nodes.control.node[0]);
            Postoffice::Get()->UpdateHeartbeat(recovery_nodes.control.node[0].id, t);
            Message back;
            for (int r : Postoffice::Get()->GetNodeIDs(
                     kWorkerGroup + kServerGroup)) {
              if (r != recovery_nodes.control.node[0].id
                    && dead_set.find(r) != dead_set.end()) {
                // do not try to send anything to dead node
                continue;
              }
              // only send recovery_node to nodes already exist
              // but send all nodes to the recovery_node
              back.meta = (r == recovery_nodes.control.node[0].id) ? nodes : recovery_nodes;
              back.meta.recver = r;
              back.meta.timestamp = timestamp_++;
              Send(back);
            }
          }
        } else {
          for (const auto& node : ctrl.node) {
            Connect(node);
            if (!node.is_recovery && node.role == Node::SERVER) ++num_servers_;
            if (!node.is_recovery && node.role == Node::WORKER) ++num_workers_;
          }
          PS_VLOG(1) << my_node_.ShortDebugString() << " is connected to others";
          ready_ = true;
        }
      } else if (ctrl.cmd == Control::BARRIER) {
        if (msg.meta.request) {
          if (barrier_count_.empty()) {
            barrier_count_.resize(8, 0);
          }
          int group = ctrl.barrier_group;
          ++barrier_count_[group];
          PS_VLOG(1) << "Barrier count for " << group << " : " << barrier_count_[group];
          if (barrier_count_[group] ==
              static_cast<int>(Postoffice::Get()->GetNodeIDs(group).size())) {
            barrier_count_[group] = 0;
            Message res;
            res.meta.request = false;
            res.meta.control.cmd = Control::BARRIER;
            for (int r : Postoffice::Get()->GetNodeIDs(group)) {
              res.meta.recver = r;
              res.meta.timestamp = timestamp_++;
              CHECK_GT(Send(res), 0);
            }
          }
        } else {
          Postoffice::Get()->Manage(msg);
        }
      } else if (ctrl.cmd == Control::HEARTBEAT) {
        time_t t = time(NULL);
        for (auto &node : ctrl.node) {
          Postoffice::Get()->UpdateHeartbeat(node.id, t);
          if (is_scheduler_) {
            Message heartbeat_ack;
            heartbeat_ack.meta.recver = node.id;
            heartbeat_ack.meta.control.cmd = Control::HEARTBEAT;
            heartbeat_ack.meta.control.node.push_back(my_node_);
            heartbeat_ack.meta.timestamp = timestamp_++;
            // send back heartbeat
            Send(heartbeat_ack);
          }
        }
      }
    } else {
      CHECK_NE(msg.meta.sender, Meta::kEmpty);
      CHECK_NE(msg.meta.recver, Meta::kEmpty);
      CHECK_NE(msg.meta.customer_id, Meta::kEmpty);
      int id = msg.meta.customer_id;
      auto* obj = Postoffice::Get()->GetCustomer(id, 5);
      CHECK(obj) << "timeout (5 sec) to wait App " << id << " ready";
      obj->Accept(msg);
    }
  }
}

void Van::PackMeta(const Meta& meta, char** meta_buf, int* buf_size) {
  // convert into protobuf
  PBMeta pb;
  pb.set_head(meta.head);
  if (meta.customer_id != Meta::kEmpty) pb.set_customer_id(meta.customer_id);
  if (meta.timestamp != Meta::kEmpty) pb.set_timestamp(meta.timestamp);
  if (meta.body.size()) pb.set_body(meta.body);
  pb.set_push(meta.push);
  pb.set_request(meta.request);
  pb.set_simple_app(meta.simple_app);
  for (auto d : meta.data_type) pb.add_data_type(d);
  if (!meta.control.empty()) {
    auto ctrl = pb.mutable_control();
    ctrl->set_cmd(meta.control.cmd);
    if (meta.control.cmd == Control::BARRIER) {
      ctrl->set_barrier_group(meta.control.barrier_group);
    } else if (meta.control.cmd == Control::ACK) {
      ctrl->set_msg_sig(meta.control.msg_sig);
    }
    for (const auto& n : meta.control.node) {
      auto p = ctrl->add_node();
      p->set_id(n.id);
      p->set_role(n.role);
      p->set_port(n.port);
      p->set_hostname(n.hostname);
      p->set_is_recovery(n.is_recovery);
    }
  }

  // to string
  *buf_size = pb.ByteSize();
  *meta_buf = new char[*buf_size+1];
  CHECK(pb.SerializeToArray(*meta_buf, *buf_size))
      << "failed to serialize protbuf";
}

void Van::UnpackMeta(const char* meta_buf, int buf_size, Meta* meta) {
  // to protobuf
  PBMeta pb;
  CHECK(pb.ParseFromArray(meta_buf, buf_size))
      << "failed to parse string into protobuf";

  // to meta
  meta->head = pb.head();
  meta->customer_id = pb.has_customer_id() ? pb.customer_id() : Meta::kEmpty;
  meta->timestamp = pb.has_timestamp() ? pb.timestamp() : Meta::kEmpty;
  meta->request = pb.request();
  meta->push = pb.push();
  meta->simple_app = pb.simple_app();
  meta->body = pb.body();
  meta->data_type.resize(pb.data_type_size());
  for (int i = 0; i < pb.data_type_size(); ++i) {
    meta->data_type[i] = static_cast<DataType>(pb.data_type(i));
  }
  if (pb.has_control()) {
    const auto& ctrl = pb.control();
    meta->control.cmd = static_cast<Control::Command>(ctrl.cmd());
    meta->control.barrier_group = ctrl.barrier_group();
    meta->control.msg_sig = ctrl.msg_sig();
    for (int i = 0; i < ctrl.node_size(); ++i) {
      const auto& p = ctrl.node(i);
      Node n;
      n.role = static_cast<Node::Role>(p.role());
      n.port = p.port();
      n.hostname = p.hostname();
      n.id = p.has_id() ? p.id() : Node::kEmpty;
      n.is_recovery = p.is_recovery();
      meta->control.node.push_back(n);
    }
  } else {
    meta->control.cmd = Control::EMPTY;
  }
}

void Van::Heartbeat() {
  const char* val = Environment::Get()->find("PS_HEARTBEAT_INTERVAL");
  const int interval = val ? atoi(val) : kDefaultHeartbeatInterval;
  while (interval > 0 && ready_) {
    std::this_thread::sleep_for(std::chrono::seconds(interval));
    Message msg;
    msg.meta.recver = kScheduler;
    msg.meta.control.cmd = Control::HEARTBEAT;
    msg.meta.control.node.push_back(my_node_);
    msg.meta.timestamp = timestamp_++;
    Send(msg);
  }
}
}  // namespace ps
