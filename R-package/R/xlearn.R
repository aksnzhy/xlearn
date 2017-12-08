# Simple interface for training an xlearn model.
#
#' @rdname xlearn
#' @export
xlearn = function(params = list(), data, label,
                  type = c("linear", "fm", "ffm"),
                  validate = NULL, model.path = "./model.out") {
    handle = xl.create(type)
    
    xl.set.train(handle, data, label)
    if (!is.null(validate)) {
        xl.set.validate(handle, validate)
    }
    xl.set.param(handle, params)
    xl.fit(handle, model.path)
    return(handle)
}

#' Predict method for xlearn model
#'
#' @rdname xlearn
#' @export
predict.xl.model = function(object, newdata, out.path = "./pred.out") {
    handle = object$handle
    model.path = object$model.path
    
    .Call(XLearnPredict_R, handle, model.path, out.path, PACKAGE = "xlearn")
    
    pred.val = readLines(out.path)
    return(pred.val)
}

# Various imports
#' @import methods
#' @useDynLib xlearn, .registration = TRUE
NULL

