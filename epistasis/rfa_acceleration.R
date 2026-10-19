rfa_lbfgs <- function(call_eval, call_grad, vars, environment = new.env(), ...,
                      invisible = 0, m = 6, epsilon = 1e-5, past = 0, delta = 0,
                      max_iterations = 0, linesearch_algorithm = "LBFGS_LINESEARCH_DEFAULT",
                      max_linesearch = 20, min_step = 1e-20, max_step = 1e20,
                      ftol = 1e-4, wolfe = 0.9, gtol = 0.9,
                      orthantwise_c = 0, orthantwise_start = 0,
                      orthantwise_end = length(vars)) {
  stopifnot(is.function(call_eval) == is.function(call_grad))
  N <- length(vars)
  stopifnot(orthantwise_end <= N)
  linesearch <- switch(linesearch_algorithm,
                       LBFGS_LINESEARCH_DEFAULT = if (orthantwise_c == 0) 0 else 2,
                       LBFGS_LINESEARCH_MORETHUENTE = 0,
                       LBFGS_LINESEARCH_BACKTRACKING_ARMIJO = 1,
                       LBFGS_LINESEARCH_BACKTRACKING = 2,
                       LBFGS_LINESEARCH_BACKTRACKING_WOLFE = 2,
                       LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 3,
                       stop("Unknown L-BFGS line search algorithm"))
  stopifnot(linesearch_algorithm != "LBFGS_LINESEARCH_MORETHUENTE" || orthantwise_c == 0)
  result <- .Call("RlibLBFGS_lbfgs", PACKAGE = "lbfgs", call_eval, call_grad, vars,
                  environment, N, invisible, m, epsilon, past, delta, max_iterations,
                  linesearch, max_linesearch, min_step, max_step, ftol, wolfe, gtol,
                  .Machine$double.eps, orthantwise_c, orthantwise_start, orthantwise_end)
  list(value = result[1L], par = result[seq_len(N) + 1L], convergence = result[N + 2L])
}

enable_rfa_acceleration <- function(ledger_path, optimizer_controls = NULL) {
  author_constructor <- construct_phenotype_operator
  construct_phenotype_operator <<- function(genotypes, site_combn) {
    as(author_constructor(genotypes, site_combn), "dMatrix")
  }
  gradient <<- function(param) {
    if (nonspec_type == "none") {
      -2 * as.vector(crossprod(X_active, z0))
    } else if (nonspec_type == "sigmoid") {
      nonspec_param <- tail(param, 2L)
      z1 <- -2 * (y_active - nonspec_param[1L] - z0)
      z2 <- z1 * z0 / nonspec_param[2L]
      z3 <- z2 * (nonspec_param[2L] - z0)
      c(as.vector(crossprod(X_active, z3)), sum(z1), sum(z2))
    } else {
      stop("Unsupported nonspecific epistasis function")
    }
  }
  .GlobalEnv$lbfgs <- function(call_eval, call_grad, vars, ...) {
    phase <- if (identical(call_eval, objective)) "fit" else "normalization"
    options <- list(...)
    if (!is.null(optimizer_controls)) {
      options$max_iterations <- if (phase == "fit") optimizer_controls$max_iterations else optimizer_controls$normalization_max_iterations
      options$past <- if (phase == "fit") optimizer_controls$past else optimizer_controls$normalization_past
      options$delta <- if (phase == "fit") optimizer_controls$delta else optimizer_controls$normalization_delta
      stopifnot(options$max_iterations > 0L, options$past >= 0L, options$delta >= 0)
    }
    calls <- 0L
    started <- proc.time()[3L]
    last_report <- started
    monitored_objective <- function(param) {
      calls <<- calls + 1L
      value <- call_eval(param)
      now <- proc.time()[3L]
      if (calls == 1L || now - last_report >= 30) {
        cat("Objective call", calls, "value", format(value, digits = 15),
            "parameter_l2", sqrt(sum(param ^ 2)),
            "gradient_l2", sqrt(sum(call_grad(param) ^ 2)),
            "elapsed_seconds", now - started, "\n")
        last_report <<- now
        flush.console()
      }
      value
    }
    cat("Starting solver with", length(vars), "parameters\n")
    flush.console()
    result <- do.call(rfa_lbfgs, c(list(call_eval = monitored_objective, call_grad = call_grad, vars = vars), options))
    reason <- if (result$convergence == -997) "iteration_limit" else if (result$convergence == 1) "relative_objective_tolerance" else if (result$convergence %in% c(0, 2)) "gradient_tolerance" else paste0("solver_status_", result$convergence)
    record <- data.frame(parameters = length(vars), objective_calls = calls,
                          objective = result$value, convergence = result$convergence,
                          elapsed_seconds = proc.time()[3L] - started)
    if (!is.null(optimizer_controls)) {
      record <- cbind(record, data.frame(phase = phase, max_iterations = options$max_iterations,
                                         past = options$past, delta = options$delta, stop_reason = reason))
    }
    exists <- file.exists(ledger_path)
    write.table(record, ledger_path, row.names = FALSE, col.names = !exists,
                sep = ",", append = exists)
    cat("Solver returned", result$convergence, "after", calls, "objective calls in",
        record$elapsed_seconds, "seconds\n")
    flush.console()
    result
  }
}
