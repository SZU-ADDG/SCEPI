args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
directory <- normalizePath(args[1])
output <- args[2]
dir.create(output, recursive = TRUE, showWarnings = FALSE)
source("external/RFA/Scripts/RFA-functions.R")
genotypes <- unname(as.matrix(read.csv(file.path(directory, "genotypes.csv"), header = FALSE)))
storage.mode(genotypes) <- "integer"
n <- ncol(genotypes)
q_list <- apply(genotypes, 2, max)
points <- read.csv(file.path(directory, "points.csv"))$point
permission <- read.csv(file.path(directory, "row_permissions.csv"))
training <- permission$row_index[permission$role == "train"]
y <- (points[training] - mean(points[training])) / sd(points[training])
nonspec_type <- "sigmoid"
site_combn <- generate_site_combn(2)
cat("Training genotypes:", length(training), "; positions:", n, "\n")
Rprof(file.path(output, "profile.out"), interval = 0.02)
setTimeLimit(elapsed = 60, transient = FALSE)
model <- infer_model(genotypes[training, , drop = FALSE], y, site_combn,
                     lambda = 0, norm_e = TRUE, silent = 1L, epsilon = 1e-5)
Rprof(NULL)
saveRDS(model, file.path(output, "diagnostic_model.rds"))
