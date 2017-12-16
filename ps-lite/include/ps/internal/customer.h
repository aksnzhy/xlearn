/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef PS_INTERNAL_CUSTOMER_H_
#define PS_INTERNAL_CUSTOMER_H_
#include <mutex>
#include <vector>
#include <utility>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <thread>
#include <memory>
#include "ps/internal/message.h"
#include "ps/internal/threadsafe_queue.h"
namespace ps {

/**
 * \brief The object for communication.
 *
 * As a sender, a customer tracks the responses for each request sent.
 *
 * It has its own receiving thread which is able to process any message received
 * from a remote node with `msg.meta.customer_id` equal to this customer's id
 */
class Customer {
 public:
  /**
   * \brief the handle for a received message
   * \param recved the received message
   */
  using RecvHandle = std::function<void(const Message& recved)>;

  /**
   * \brief constructor
   * \param id the unique id, any received message with
   * \param recv_handle the functino for processing a received message
   */
  Customer(int id, const RecvHandle& recv_handle);

  /**
   * \brief desconstructor
   */
  ~Customer();

  /**
   * \brief return the unique id
   */
  int id() { return id_; }

  /**
   * \brief get a timestamp for a new request. threadsafe
   * \param recver the receive node id of this request
   * \return the timestamp of this request
   */
  int NewRequest(int recver);


  /**
   * \brief wait until the request is finished. threadsafe
   * \param timestamp the timestamp of the request
   */
  void WaitRequest(int timestamp);

  /**
   * \brief return the number of responses received for the request. threadsafe
   * \param timestamp the timestamp of the request
   */
  int NumResponse(int timestamp);

  /**
   * \brief add a number of responses to timestamp
   */
  void AddResponse(int timestamp, int num = 1);

  /**
   * \brief accept a received message from \ref Van. threadsafe
   * \param recved the received the message
   */
  void Accept(const Message& recved) { recv_queue_.Push(recved); }

 private:
  /**
   * \brief the thread function
   */
  void Receiving();

  int id_;

  RecvHandle recv_handle_;
  ThreadsafeQueue<Message> recv_queue_;
  std::unique_ptr<std::thread> recv_thread_;

  std::mutex tracker_mu_;
  std::condition_variable tracker_cond_;
  std::vector<std::pair<int, int>> tracker_;

  DISALLOW_COPY_AND_ASSIGN(Customer);
};

}  // namespace ps
#endif  // PS_INTERNAL_CUSTOMER_H_
