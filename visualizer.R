#!/usr/bin/env Rscript
#
# kg_visualizer.R — an R/kgraph re‑implementation of KG_visualizer.py
#
# Features (parity with your Python version when possible):
# - Loads JSON or JSONL graphs (D3-style nodes/edges, triples, or line‑wise triples)
# - Filters edges by min weight
# - Keeps largest connected component (optional)
# - k‑core pruning (via igraph coreness)
# - Caps node count by top degree
# - Labels top‑K nodes by degree
# - Highlights the top‑K strongest edges
# - Multiple layouts (spring/FR, KK, circle, random, etc.)
# - Interactive visualization via kgraph + sgraph (Sigma.js)
#
# Output: an interactive HTML widget (and optional PNG snapshot via webshot2)
#

suppressPackageStartupMessages({
  library(jsonlite)
  library(data.table)
  library(optparse)
  library(igraph)
  library(kgraph)
  library(sgraph)
  library(htmlwidgets)
})

# ---------------- CLI ----------------
option_list <- list(
  make_option("--input", type = "character", default = "graph.json",
              help = "Path to graph JSON/JSONL"),
  make_option("--out", type = "character", default = "graph_viz.html",
              help = "Output HTML path (interactive)"),
  make_option("--png", type = "character", default = NA,
              help = "Optional PNG snapshot path (requires webshot2)"),
  make_option("--min-weight", type = "double", default = 2.0,
              help = "Keep edges with weight >= this"),
  make_option("--largest-only", action = "store_true", default = FALSE,
              help = "Keep only largest connected component"),
  make_option("--k-core", type = "integer", default = 0,
              help = "If >0, take k-core of the (sub)graph"),
  make_option("--max-nodes", type = "integer", default = 300,
              help = "Cap node count by degree to avoid clutter"),
  make_option("--label-top", type = "integer", default = 20,
              help = "Label top-K nodes by degree"),
  make_option("--seed", type = "integer", default = 42,
              help = "Layout seed"),
  make_option("--highlight-edges-top", type = "integer", default = 20,
              help = "Highlight top-K strongest edges"),
  make_option("--edge-labels-top", type = "integer", default = 0,
              help = "(Static only) show labels for top-K edges — used if PNG snapshot is requested"),
  make_option("--layout", type = "character", default = "spring",
              help = "Layout: spring, kk, circular, random, sphere, lgl, fr, grid, drl")
)

opt <- parse_args(OptionParser(option_list = option_list))
set.seed(opt$seed)

in_path <- opt$input
if (!file.exists(in_path)) stop(sprintf("Missing input: %s", in_path))

# ------------- Robust loader -------------
# Returns a list with: nodes (character vector), edges (data.table u,v,w,label)
load_graph <- function(path) {
  rel_bag <- new.env(parent = emptyenv())  # key "u\tv" -> named numeric vector of relations

  add_rel <- function(u, v, r, w) {
    key <- paste0(u, "\t", v)
    cur <- get0(key, envir = rel_bag, ifnotfound = numeric())
    if (!is.null(r) && nzchar(r)) {
      if (is.na(w) || !is.finite(w)) w <- 1
      cur[r] <- (cur[r] %||% 0) + w
      assign(key, cur, envir = rel_bag)
    }
  }

  edges <- data.table(u = character(), v = character(), w = numeric(), label = character())
  nodes_set <- new.env(parent = emptyenv())
  add_node <- function(x) assign(as.character(x), TRUE, envir = nodes_set)

  # JSONL triples
  if (tolower(tools::file_ext(path)) == "jsonl") {
    con <- file(path, open = "r", encoding = "UTF-8")
    on.exit(close(con), add = TRUE)
    repeat {
      line <- readLines(con, n = 1, warn = FALSE)
      if (length(line) == 0) break
      line <- trimws(line)
      if (identical(line, "")) next
      rec <- tryCatch(jsonlite::fromJSON(line), error = function(e) NULL)
      if (is.null(rec)) next
      h <- rec$h %||% rec$subject
      t <- rec$t %||% rec$object
      r <- rec$r %||% rec$relation %||% ""
      w <- rec$weight %||% 1
      if (is.null(h) || is.null(t)) next
      u <- as.character(h); v <- as.character(t); w <- suppressWarnings(as.numeric(w)); if (!is.finite(w)) w <- 1
      edges <- rbind(edges, data.table(u = u, v = v, w = w, label = NA_character_), fill = TRUE)
      add_node(u); add_node(v)
      add_rel(u, v, r, w)
    }
  } else {
    dat <- jsonlite::fromJSON(path, simplifyVector = TRUE)

    # Triples format: { triples: [ {h/r/t/weight} ] }
    if (is.list(dat) && !is.null(dat$triples)) {
      for (i in seq_len(nrow(as.data.frame(dat$triples)))) {
        tri <- dat$triples[i, ]
        h <- tri$h %||% tri$subject
        t <- tri$t %||% tri$object
        r <- tri$r %||% tri$relation %||% ""
        w <- tri$weight %||% 1
        if (is.null(h) || is.null(t)) next
        u <- as.character(h); v <- as.character(t); w <- suppressWarnings(as.numeric(w)); if (!is.finite(w)) w <- 1
        edges <- rbind(edges, data.table(u = u, v = v, w = w, label = NA_character_), fill = TRUE)
        add_node(u); add_node(v)
        add_rel(u, v, r, w)
      }
    } else {
      # D3/list-of-edges schemas
      nodes <- dat$nodes %||% dat$Vertices %||% NULL
      links <- dat$edges %||% dat$links %||% dat$Edges %||% dat

      nid <- function(n) {
        if (is.list(n)) as.character(n$id %||% n$name %||% n$label) else as.character(n)
      }
      if (!is.null(nodes)) {
        for (i in seq_along(nodes)) {
          x <- tryCatch(nid(nodes[[i]]), error = function(e) NA_character_)
          if (!is.na(x)) add_node(x)
        }
      }
      if (is.data.frame(links) || is.list(links)) {
        if (is.data.frame(links)) {
          L <- split(links, seq_len(nrow(links)))
        } else {
          L <- links
        }
        for (e in L) {
          if (is.list(e)) {
            u <- e$source %||% e$from %||% e$u
            v <- e$target %||% e$to %||% e$v
            w <- e$weight %||% 1
          } else if (is.atomic(e) && length(e) >= 2) {
            u <- e[[1]]; v <- e[[2]]; w <- if (length(e) >= 3) e[[3]] else 1
          } else next
          if (is.null(u) || is.null(v)) next
          u <- as.character(u); v <- as.character(v); w <- suppressWarnings(as.numeric(w)); if (!is.finite(w)) w <- 1
          edges <- rbind(edges, data.table(u = u, v = v, w = w, label = NA_character_), fill = TRUE)
          add_node(u); add_node(v)
        }
      }
    }
  }

  # attach a dominant relation label if present
  if (nrow(edges)) {
    keys <- paste0(edges$u, "\t", edges$v)
    rels <- vapply(keys, function(k) {
      vec <- get0(k, envir = rel_bag, ifnotfound = numeric())
      if (length(vec)) names(vec)[which.max(vec)] else NA_character_
    }, character(1))
    edges$label <- rels
  }

  nodes <- ls(envir = nodes_set, all.names = FALSE)
  list(nodes = nodes, edges = edges)
}

`%||%` <- function(a, b) if (!is.null(a)) a else b

loaded <- load_graph(in_path)
edges <- loaded$edges
if (!nrow(edges)) stop("Graph is empty or could not be parsed.")

# Threshold by weight
edges <- edges[w >= opt$`min-weight`]
if (!nrow(edges)) stop("Graph is empty after filtering. Try lowering --min-weight.")

# Build an igraph from edges for degree calcs and pruning
ig <- graph_from_data_frame(d = edges[, .(from = u, to = v, weight = w)], directed = FALSE)

# Remove isolates (post-threshold)
ig <- delete_vertices(ig, V(ig)[degree(ig) == 0])
if (vcount(ig) == 0) stop("Graph is empty after removing isolates.")

# Largest component (optional)
if (isTRUE(opt$`largest-only`) && components(ig)$no > 1) {
  comp <- components(ig)
  keep <- which.max(comp$csize)
  ig <- induced_subgraph(ig, vids = V(ig)[comp$membership == keep])
}

# k-core (optional)
if (opt$`k-core` > 0) {
  core <- coreness(ig)
  ig <- induced_subgraph(ig, vids = V(ig)[core >= opt$`k-core`])
}

# Cap to max-nodes by top degree
if (vcount(ig) > opt$`max-nodes`) {
  deg <- degree(ig)
  keep_names <- names(sort(deg, decreasing = TRUE))[seq_len(opt$`max-nodes`)]
  ig <- induced_subgraph(ig, vids = V(ig)[name %in% keep_names])
}

# Label top-K by degree (store in vertex attribute 'label')
V(ig)$label <- NA_character_
if (vcount(ig) > 0 && opt$`label-top` > 0) {
  deg <- degree(ig)
  lab_names <- names(sort(deg, decreasing = TRUE))[seq_len(min(opt$`label-top`, length(deg)))]
  V(ig)$label[V(ig)$name %in% lab_names] <- V(ig)$name[V(ig)$name %in% lab_names]
}

# Highlight top-K strongest edges: increase width and set color
E(ig)$width <- 1
E(ig)$color <- "#999999"
if (ecount(ig) > 0 && opt$`highlight-edges-top` > 0) {
  ord <- order(E(ig)$weight, decreasing = TRUE)
  k <- min(opt$`highlight-edges-top`, length(ord))
  idx <- ord[seq_len(k)]
  E(ig)$width[idx] <- 3
  E(ig)$color[idx] <- "#333333"
}

# Edge labels for top‑K (only used if PNG requested; sgraph/Sigma rarely displays edge labels)
E(ig)$label <- NA_character_
if (!is.na(opt$png) && opt$`edge-labels-top` > 0 && ecount(ig) > 0 && !is.null(edges$label)) {
  ord <- order(E(ig)$weight, decreasing = TRUE)
  k <- min(opt$`edge-labels-top`, length(ord))
  idx <- ord[seq_len(k)]
  # Map back to our loaded labels using endpoints
  el <- as.data.table(as_edgelist(ig, names = TRUE))
  setnames(el, c("V1","V2"), c("u","v"))
  el[, key := paste0(u, "\t", v)]
  edges[, key := paste0(u, "\t", v)]
  labmap <- edges[!is.na(label), .(label = label[.N]), by = key]
  labs <- labmap$label[match(el$key, labmap$key)]
  E(ig)$label <- NA_character_
  E(ig)$label[idx] <- labs[idx]
}

# Choose layout
layout_fun <- switch(tolower(opt$layout),
  "spring" = layout_with_fr,
  "fr"     = layout_with_fr,
  "kk"     = layout_with_kk,
  "kamada_kawai" = layout_with_kk,
  "circular" = layout_in_circle,
  "random" = layout_randomly,
  "sphere" = layout_on_sphere,
  "lgl" = layout_with_lgl,
  "grid" = layout_on_grid,
  "drl" = layout_with_drl,
  layout_with_fr
)

# ---- Build a kgraph l_graph object so we stay in the kgraph+sgraph ecosystem ----
# Node table: id, desc (use id as desc by default)
df_nodes <- data.frame(id = V(ig)$name, desc = ifelse(is.na(V(ig)$label), V(ig)$name, V(ig)$label), stringsAsFactors = FALSE)
# Edge table: source/target/weight; sgraph expects names 'source','target','weight'
el <- as.data.frame(get.edges(ig, E(ig)))
colnames(el) <- c("source","target")
df_links <- data.frame(source = V(ig)$name[el$source], target = V(ig)$name[el$target], weight = E(ig)$weight, stringsAsFactors = FALSE)

kg_obj <- list(df_nodes = df_nodes, df_links = df_links)  # same structure that build_kgraph() returns
class(kg_obj) <- c("l_graph", class(kg_obj))

# Convert to igraph through sgraph helper (adds df_nodes as vertex attrs)
ig2 <- sgraph::l_graph_to_igraph(kg_obj)
# Bring styling attributes we computed on ig
V(ig2)$label <- V(ig)$label
E(ig2)$width <- E(ig)$width
E(ig2)$color <- E(ig)$color

# Build sgraph widget with chosen layout
sg <- sgraph::sgraph_clusters(
  ig2,
  node_size = NULL,          # node size already fine; could map to degree or strength if desired
  label = "label",
  layout = layout_fun(ig2)
)

# Save interactive HTML
saveWidget(sg, file = opt$out, selfcontained = TRUE)
cat(sprintf("Saved interactive graph to %s\n", opt$out))

# Optional PNG snapshot (static), including edge labels if requested
if (!is.na(opt$png)) {
  if (!requireNamespace("webshot2", quietly = TRUE)) {
    warning("Install 'webshot2' for PNG snapshots: install.packages('webshot2')")
  } else {
    webshot2::webshot(opt$out, file = opt$png, vwidth = 1400, vheight = 900)
    cat(sprintf("Saved PNG snapshot to %s\n", opt$png))
  }
}

