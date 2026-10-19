args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
root <- normalizePath(".")
directory <- normalizePath(args[1])
set.seed(as.integer(args[2]))
source(file.path(root, "external/RFA/Scripts/RFA-functions.R"))
policy_path <- Sys.getenv("RECOMB_RFA_POLICY")
bounded <- nzchar(policy_path)
optimizer_controls <- if (bounded) jsonlite::read_json(policy_path, simplifyVector = TRUE) else NULL
fixed_default <- bounded && optimizer_controls$penalty_candidates == 1
accelerated <- bounded || identical(Sys.getenv("RECOMB_RFA_ACCELERATION"), "1")
if (accelerated) {
  source(file.path(root, "epistasis/rfa_acceleration.R"))
  enable_rfa_acceleration(file.path(directory, "solver_events.csv"), optimizer_controls)
  dir.create(file.path(directory, "trial_models"), showWarnings = FALSE)
}
if (bounded) {
  stopifnot(optimizer_controls$max_iterations > 0, optimizer_controls$penalty_candidates %in% c(1, 12))
  if (fixed_default) stopifnot(optimizer_controls$lambda_per_observation == 0)
  jsonlite::write_json(optimizer_controls, file.path(directory, "optimizer_policy.json"), auto_unbox = TRUE, pretty = TRUE)
}
genotypes <- unname(as.matrix(read.csv(file.path(directory, "genotypes.csv"), header = FALSE)))
storage.mode(genotypes) <- "integer"
n <- ncol(genotypes)
q_list <- apply(genotypes, 2, max)
stopifnot(n >= 2, all(q_list >= 2))
points <- read.csv(file.path(directory, "points.csv"))$point
permission <- read.csv(file.path(directory, "row_permissions.csv"))
training <- permission$row_index[permission$role == "train"]
validation <- permission$row_index[permission$role == "validation"]
stopifnot(length(intersect(training, validation)) == 0)
stopifnot(all(is.finite(points[training])), all(is.finite(points[validation])))
center <- mean(points[training])
scale <- sd(points[training])
stopifnot(is.finite(scale), scale > 0)
y <- (points - center) / scale
nonspec_type <- "sigmoid"
site_combn <- generate_site_combn(2)
epsilon_path <- file.path(directory, "validation_quartets.csv")
use_epsilon <- file.exists(epsilon_path)
if (use_epsilon) {
  quartets <- read.csv(epsilon_path)
  stopifnot(all(quartets$ab %in% validation))
  evaluation <- sort(unique(unlist(quartets[, c("wt", "a", "b", "ab")])))
} else {
  evaluation <- validation
}
operator <- construct_phenotype_operator(genotypes[evaluation, , drop = FALSE], site_combn)
penalties <- if (fixed_default) length(training) * optimizer_controls$lambda_per_observation else
  length(training) * c(0, 10 ^ seq(-6, -1, length.out = 11))
score_model <- function(fitted) {
  stopifnot(all(is.finite(fitted$e)), all(is.finite(fitted$nonspec_param)))
  prediction <- apply_nonspecific_epistasis(as.vector(operator %*% fitted$e), fitted$nonspec_param) * scale + center
  stopifnot(all(is.finite(prediction)))
  names(prediction) <- evaluation
  if (use_epsilon) {
    predicted_epsilon <- prediction[as.character(quartets$ab)] - prediction[as.character(quartets$a)] -
      prediction[as.character(quartets$b)] + prediction[as.character(quartets$wt)]
    primary <- sqrt(mean((predicted_epsilon - quartets$epsilon) ^ 2))
    list(primary = primary, secondary = primary, rho = NA_real_)
  } else {
    rho <- cor(points[validation], prediction, method = "spearman")
    list(primary = if (is.finite(rho)) -rho else Inf,
         secondary = sqrt(mean((points[validation] - prediction) ^ 2)), rho = rho)
  }
}
prior_models <- vector("list", length(penalties))
if (bounded && file.exists(file.path(directory, "prior_sources.json"))) {
  sources <- jsonlite::read_json(file.path(directory, "prior_sources.json"), simplifyVector = TRUE)
  for (prior in sources) {
    checkpoints <- file.path(prior, "trial_models", paste0("trial", seq_along(penalties), ".rds"))
    best_path <- file.path(prior, "model.rds")
    if (!any(file.exists(checkpoints)) && !file.exists(best_path)) next
    previous_genotypes <- unname(as.matrix(read.csv(file.path(prior, "genotypes.csv"), header = FALSE)))
    storage.mode(previous_genotypes) <- "integer"
    previous_permission <- read.csv(file.path(prior, "row_permissions.csv"))
    previous_points <- read.csv(file.path(prior, "points.csv"))$point
    stopifnot(identical(previous_genotypes, genotypes), identical(previous_permission, permission),
              identical(previous_points[c(training, validation)], points[c(training, validation)]))
    for (index in which(file.exists(checkpoints))) {
      saved <- readRDS(checkpoints[index])
      fitted <- if (!is.null(saved$provenance)) saved$model else saved
      prior_models[[index]] <- c(prior_models[[index]], list(list(model = fitted, source = checkpoints[index])))
    }
    if (file.exists(best_path)) {
      previous <- readRDS(best_path)
      stopifnot(identical(previous$training_rows, training), identical(previous$validation_rows, validation),
                identical(previous$q_list, q_list), identical(previous$phenotype_center, center), identical(previous$phenotype_scale, scale))
      index <- which(abs(penalties - previous$lambda) <= 1e-12 * max(1, abs(previous$lambda)))
      stopifnot(length(index) == 1L || fixed_default)
      if (length(index) == 1L && !file.exists(checkpoints[index])) {
        prior_models[[index]] <- c(prior_models[[index]], list(list(model = previous$model, source = best_path)))
      }
    }
  }
}
history <- data.frame()
trial_sources <- vector("list", length(penalties))
best_primary <- Inf
best_secondary <- Inf
best <- NULL
for (trial_index in seq_along(penalties)) {
  penalty <- penalties[trial_index]
  checkpoint <- file.path(directory, "trial_models", paste0("trial", trial_index, ".rds"))
  if (accelerated && file.exists(checkpoint)) {
    saved <- readRDS(checkpoint)
    fitted <- if (bounded) saved$model else saved
    provenance <- if (bounded) saved$provenance else NULL
  } else {
    candidates <- prior_models[[trial_index]]
    if (bounded && length(candidates)) {
      scores <- lapply(candidates, function(candidate) score_model(candidate$model))
      selected <- order(vapply(scores, function(value) value$primary, numeric(1L)),
                        vapply(scores, function(value) value$secondary, numeric(1L)))[1L]
      fitted <- candidates[[selected]]$model
      provenance <- list(kind = "retained_completed_trial", source = candidates[[selected]]$source)
      cat("Retained completed trial", trial_index, "from", provenance$source, "\n")
    } else {
      fitted <- infer_model(genotypes[training, , drop = FALSE], y[training], site_combn,
                            lambda = penalty, norm_e = TRUE, silent = 1L,
                            epsilon = if (bounded) optimizer_controls$epsilon else 1e-5)
      provenance <- if (bounded) list(kind = "bounded_fit", policy = optimizer_controls$version) else NULL
    }
    stopifnot(all(is.finite(fitted$e)), all(is.finite(fitted$nonspec_param)))
    if (accelerated) {
      saveRDS(if (bounded) list(model = fitted, provenance = provenance) else fitted, paste0(checkpoint, ".pending"))
      stopifnot(file.rename(paste0(checkpoint, ".pending"), checkpoint))
    }
  }
  score <- score_model(fitted)
  primary <- score$primary
  secondary <- score$secondary
  rho <- score$rho
  if (bounded) {
    trial_sources[[trial_index]] <- provenance
    jsonlite::write_json(trial_sources, file.path(directory, "trial_sources.json"), auto_unbox = TRUE, pretty = TRUE)
  }
  history <- rbind(history, data.frame(lambda = penalty, lambda_per_observation = penalty / length(training),
                                      primary_score = primary, secondary_score = secondary, validation_spearman = rho))
  write.csv(history, file.path(directory, "trials.csv"), row.names = FALSE)
  if (primary < best_primary || (primary == best_primary && secondary < best_secondary)) {
    best <- fitted
    best_primary <- primary
    best_secondary <- secondary
    saveRDS(list(model = fitted, phenotype_center = center, phenotype_scale = scale, q_list = q_list,
                 site_combn = site_combn, lambda = penalty, training_rows = training,
                 validation_rows = validation), file.path(directory, "model.rds"))
  }
  cat(sprintf("Completed lambda %.8g; validation score %.8g\n", penalty, primary))
}
stopifnot(!is.null(best))
predictions <- numeric(nrow(genotypes))
for (start in seq.int(1L, nrow(genotypes), by = 256L)) {
  rows <- start:min(start + 255L, nrow(genotypes))
  current <- construct_phenotype_operator(genotypes[rows, , drop = FALSE], site_combn)
  predictions[rows] <- apply_nonspecific_epistasis(as.vector(current %*% best$e), best$nonspec_param) * scale + center
}
stopifnot(all(is.finite(predictions)))
write.csv(data.frame(row_index = seq_along(predictions), prediction = predictions),
          file.path(directory, "raw_predictions.csv"), row.names = FALSE)
write.csv(data.frame(training_variants = length(training), validation_variants = length(validation),
                     genotype_positions = n, coefficients = length(best$e), phenotype_center = center,
                     phenotype_scale = scale, trials = length(penalties), epsilon_selection = use_epsilon),
          file.path(directory, "fit_summary.csv"), row.names = FALSE)
if (bounded) {
  counts <- table(vapply(trial_sources, function(value) value$kind, character(1L)))
  jsonlite::write_json(list(complete = TRUE, policy = optimizer_controls,
                           newly_fitted_trials = sum(vapply(trial_sources, function(value) value$kind == "bounded_fit", logical(1L))),
                           retained_trials = sum(vapply(trial_sources, function(value) value$kind == "retained_completed_trial", logical(1L)))),
                      file.path(directory, "budget_summary.json"), auto_unbox = TRUE, pretty = TRUE)
}
