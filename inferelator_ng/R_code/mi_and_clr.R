library("inline")
if ('parallel' %in% installed.packages()[, 'Package']) {
  library('parallel')
} else {
  library('multicore')
}

discretize <- function(X, nbins) {
  N <- length(X)
  X.min <- min(X)
  X.max <- max(X)
  tiny <- max(.Machine$double.eps * (X.max - X.min), .Machine$double.eps)
  X.disc <- floor((X - X.min) / (X.max - X.min + tiny) * nbins)
  return(as.integer(X.disc))
}


# compute MI of each colum in x to each column in y
# you'll want the larger matrix to be x
mi <- function(x, y, nbins=10, cpu.n=1, perm.mat=NULL) {
  if(!is.null(perm.mat)) {
    for (i in 1:ncol(x)) {
      x[, i] <- x[perm.mat[, i], i]
    }
    for (i in 1:ncol(y)) {
      y[, i] <- y[perm.mat[, i], i]
    }
  }
  
  # discretize the columns
  x <- apply(x, 2, discretize, nbins)
  y <- apply(y, 2, discretize, nbins)
  
  m <- ncol(x)
  n <- ncol(y)
  ret <- matrix(0, m, n, dimnames=list(colnames(x), colnames(y)))
  
  if (cpu.n == 1) {  
    s <- nrow(x)
    return(matrix(mi.in.c4(s, nbins, x, m, y, n, ret)$mi, m, dimnames=list(colnames(x), colnames(y))))
  }
  
  col.list <- split(1:m, cut(1:m, cpu.n, labels=F))
  ret.list <- mclapply(col.list, mi.thread, x, y, nbins, mc.cores = cpu.n)
  for (i in 1:length(ret.list)) {
    ret[col.list[[i]], ] <- ret.list[[i]]
  }
  return(ret)
}


mi.thread <- function(cols, x, y, nbins) {
  x <- x[, cols, drop = FALSE]
  m <- ncol(x)
  n <- ncol(y)
  s <- nrow(x)
  ret <- matrix(0, m, n)
  return(matrix(mi.in.c4(s, nbins, x, m, y, n, ret)$mi, m))
}

mi.in.c4.sig <- signature(n = 'integer', k = 'integer', x = 'integer', 
                          ncolx = 'integer', y = 'integer', 
                          ncoly = 'integer', mi = 'numeric')
mi.in.c4.code <- '
  int i, j, cx, cy;
  double *pxy, *px, *py;

  py = (double *)calloc(*k * *ncoly, sizeof(double));
  for (cy = 0; cy < *ncoly; ++cy) {
    for (i = 0; i < *n; ++i) {
      ++py[cy * *k + y[cy * *n + i]];
    }
  }

  for (cx = 0; cx < *ncolx; ++cx) {
    for (cy = 0; cy < *ncoly; ++cy) {
      pxy = (double *)calloc(*k * *k, sizeof(double));
      px = (double *)calloc(*k, sizeof(double));

      for (i = 0; i < *n; ++i) {
        ++pxy[(y[cy * *n + i]) * (*k) + (x[cx * *n + i])];
        ++px[x[cx * *n + i]];
      }
      for (i = 0; i < *k; ++i) {
        for (j = 0; j < *k; ++j) {
          if (pxy[j * (*k) + i] > 0) {
            mi[cy * *ncolx + cx] += pxy[j * (*k) + i] / *n * 
                                    log(pxy[j * (*k) + i] / *n / 
                                       (px[i] / *n * py[cy * *k + j] / *n));
          }
        }
      }
      free(px);
      free(pxy);
    }
  }
  free(py);'
mi.in.c4 <- cfunction(mi.in.c4.sig, mi.in.c4.code, convention=".C")


toZscore <- function(x, bg = NULL) {
  if (is.null(bg) == T) {
    return((x - mean(x, na.rm = T)) / sd(x, na.rm = T))
  }
  return((x - mean(bg, na.rm = T)) / sd(bg, na.rm = T))
}


mixedCLR <- function(mi.stat, mi.dyn) {
  z.r.dyn <- t(apply(mi.dyn, 1, toZscore))
  z.r.dyn[z.r.dyn < 0] <- 0
  
  z.c.mix <- mi.dyn
  for (j in 1:ncol(mi.stat)) {
    z.c.mix[, j] <- (z.c.mix[, j] - mean(mi.stat[, j], na.rm = T)) / sd(mi.stat[, j], na.rm = T)
  }
  z.c.mix[z.c.mix < 0] <- 0
  
  mclr <- sqrt(z.r.dyn ^ 2 + z.c.mix ^ 2)
  mclr[is.nan(mclr)] <- NA
  return(mclr)
}
