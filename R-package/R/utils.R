#
# This file is for the low level reuseable utility functions
# that are not supposed to be visibe to a user.
#

xl.create = function(model) {
    if (model %in% c("linear", "fm", "ffm")) {
        handle = .Call(XLearnCreate_R, model, PACKAGE = "xlearn")
    } else {
        stop("Invalid type of model '", model, "'")
    }
    return(handle)
}

write.data.file = function(path, data, label = NULL) {
    if (is.null(label)) {
        df = data
    } else {
        df = data.frame(label, data)
    }
    write.table(df, file = path, quote = FALSE, sep = "\t",
                row.names = FALSE, col.names = FALSE)
}

xl.set.params = function(handle, params) {
    n = length(params)
    nms = names(params)
    list.str = c("task", "metric", "log", "opt")
    list.float = c("lambda", "init", "alpha", "beta",
                   "lambda_1", "lambda_2")
    list.int = c("k", "epoch", "fold")
    for (i in 1:n) {
        key = nms[i]
        value = params[[i]]
        if (key %in% list.str) {
            .Call(XLearnSetStr_R, handle, key, value, PACKAGE = "xlearn")
        } else if (key %in% list.float) {
            .Call(XLearnSetFloat_R, handle, key, value, PACKAGE = "xlearn")
        } else if (key %in% list.int) {
            .Call(XLearnSetInt_R, handle, key, value, PACKAGE = "xlearn")
        } else {
            stop(paste0("invalid parameter '", key, "'"))
        }
    }
}

# currently only read data from file
xl.set.train = function(handle, data, label) {
    tmpf = tempfile()
    write.data.file(tmpf, data, label)
    .Call(XLearnSetTrain_R, handle, tmpf, PACKAGE = "xlearn")
}

# currently only read data from file
xl.set.validate = function(handle, data, label) {
    tmpf = tempfile()
    write.data.file(tmpf, data, label)
    .Call(XLearnSetValidate_R, handle, tmpf, PACKAGE = "xlearn")
}

# currently only read data from file
xl.set.test = function(handle, data) {
    tmpf = tempfile()
    write.data.file(tmpf, data)
    .Call(XLearnSetValidate_R, handle, tmpf, PACKAGE = "xlearn")
}

# currently only save model to disk
xl.fit = function(handle, model.path) {
    .Call(XLearnFit_R, handle, model.path, PACKAGE = "xlearn")
}
