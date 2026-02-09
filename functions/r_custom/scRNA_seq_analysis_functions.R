
#### DATA WRANGLING ####

# function to identify cells in seurat objects  with certain gene expression patterns and add
# a metadata column for that subset and cell identity in that subset (pos or neg).
# Direction signifies positive or negative expression of the gene (> or <= in the 
# gene expression function) and is + by default
identify_cells_gene <- function(seurat_object,genes,cutoffs,direction=rep("+",100),slot="data"){
  # identify cells based on all gene conditions
  cellList <- vector(mode = "list", length = length(genes))
  for(i in 1:length(genes)){
    ifelse(direction[i] == "+",
           cellList[[i]] <- which(FetchData(seurat_object,genes[i],slot=slot) > cutoffs[i]),
           cellList[[i]] <- which(FetchData(seurat_object,genes[i],slot=slot) <= cutoffs[i]))
  }
  
  # create posCells vector that is intersection of all vectors in list
  if(length(cellList) > 1){
    for(j in 2:length(cellList)){
      posCells <- intersect(cellList[[1]],cellList[[j]])
    }
  }
  else{
    posCells <- cellList[[1]]
  }
  
  # create metadata column for subset
  seurat_object@meta.data[,paste0(c(rbind(genes,direction[1:length(genes)])),collapse = "")] <- ifelse(rownames(seurat_object@meta.data) %in% rownames(seurat_object@meta.data[posCells,]),"pos","neg")
  
  return(seurat_object)
}

# function to convert mouse gene names to human gene names using biomaRt
# returns a tible with MGI (mouse) symbols and matched HGNC (human) symbols
mouse_to_human_genes <- function(genelist){
  require("biomaRt")
  human <- useMart("ensembl", dataset = "hsapiens_gene_ensembl",
                   host = "https://dec2021.archive.ensembl.org/")
  mouse <- useMart("ensembl", dataset = "mmusculus_gene_ensembl",
                   host = "https://dec2021.archive.ensembl.org/")
  
  genesV2 <- getLDS(attributes = c("mgi_symbol"), filters = "mgi_symbol", values = genelist ,
                    mart = mouse, attributesL = c("hgnc_symbol"), martL = human, uniqueRows=T)
  
  genesTidy <- genesV2 %>%
    as_tibble()
  
  return(genesTidy)
}

# function to convert human gene names to mouse gene names using biomaRt
# returns a tible with MGI (mouse) symbols and matched HGNC (human) symbols
human_to_mouse_genes <- function(genelist){
  require("biomaRt")
  human <- useMart("ensembl", dataset = "hsapiens_gene_ensembl",
                   host = "https://dec2021.archive.ensembl.org/")
  mouse <- useMart("ensembl", dataset = "mmusculus_gene_ensembl",
                   host = "https://dec2021.archive.ensembl.org/")
  
  genesV2 <- getLDS(attributes = c("hgnc_symbol"), filters = "hgnc_symbol", values = genelist ,
                    mart = human, attributesL = c("mgi_symbol"), martL = mouse, uniqueRows=T)
  
  genesTidy <- genesV2 %>%
    as_tibble()
  
  return(genesTidy)
}

# Function to convert Ensembl IDs to gene names
# input is a vector of ensembl IDs and output is a vector of gene names
convert_ensembl_to_gene_names <- function(ensembl_ids) {
  # Use the ensembl dataset for mouse
  ensembl <- useMart("ensembl", dataset = "mmusculus_gene_ensembl")
  
  # Query the biomaRt database
  results <- getBM(
    attributes = c("ensembl_gene_id", "external_gene_name"),
    filters = "ensembl_gene_id",
    values = ensembl_ids,
    mart = ensembl
  )
  
  # Merge the results with the original ensembl IDs to keep the order
  ensembl_to_gene <- merge(
    data.frame(ensembl_gene_id = ensembl_ids, original_order = seq_along(ensembl_ids)),
    results,
    by = "ensembl_gene_id",
    all.x = TRUE
  )
  
  # Sort the results by the original order
  ensembl_to_gene <- ensembl_to_gene[order(ensembl_to_gene$original_order),]
  
  return(ensembl_to_gene$external_gene_name)
}

#### FEATURE PROCESSING ####

# Function to plot piecewise histograms from sparse count matrix
# Use for plotting barcodes or CRISPR guides
# Takes a row of counts from a sparse matrix of counts
histo_piecewise <- function(x, threshold = 10){
  # set histogram parameters
  threshold <- threshold
  max_count <- max(x)
  # Define linear breaks from 0 to threshold
  linear_breaks <- seq(0, threshold, by = 1)
  # Define exponential breaks beyond the threshold
  exp_breaks <- unique(c(threshold, threshold * 2^(0:ceiling(log2(max_count/threshold)))))
  # Combine the breaks
  custom_breaks <- unique(c(linear_breaks, exp_breaks))
  
  # create input dataframe
  input <- data.frame(bc_counts = as.vector(x))
  input$bin <- cut(input$bc_counts, breaks = custom_breaks, include.lowest = TRUE)
  
  # Calculate frequencies
  bin_counts <- as.data.frame(table(input$bin))
  colnames(bin_counts) <- c("bin", "count")
  bin_counts$bin <- factor(bin_counts$bin, levels = levels(input$bin))
  
  # plot histogram
  p <- ggplot(bin_counts, aes(x = bin, y = count)) +
    # add number of cells above each bar
    geom_bar(stat = "identity", fill = "steelblue", color = "black") +
    geom_text(data = bin_counts, aes(x = bin, y = count, label = count), 
              vjust = -0.5, color = "black") +
    scale_x_discrete(labels = function(x) {
      breaks_labels <- unique(c(linear_breaks, exp_breaks))
      breaks_labels[match(x, levels(input$bin))]
    }) +
    labs(x = "barcode Counts", y = "Frequency") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    theme_classic()
  
  return(p)
}

# Function to plot histogram with log10 from sparse count matrix
# Use for plotting barcodes or CRISPR guides
# Takes a row of counts from a sparse matrix of counts
histo_log <- function(x, threshold = 10){
  # create input tibble
  input <- data.frame(bc_counts = as.vector(x)) %>%
    mutate(log10_counts = log10(bc_counts+1))
  
  # plot histogram
  p <- ggplot(input, aes(x = log10_counts)) +
    # add number of cells above each bar
    geom_bar(stat = "count", fill = "steelblue", color = "black") +
    labs(x = "log10(barcode Counts+1)", y = "Frequency") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
    theme_classic()
  
  return(p)
}

# function to identify features with UMI > umi_cutoff
# returns tibble with cell barcodes and respective positive features
# used for manual annotation of CRISPR or BC calls
features_counts <- function(matrix, umi_cutoff) {
  features <- list()
  col_names <- colnames(matrix)
  for (col in 1:ncol(matrix)) {
    col_data <- matrix[, col]
    feature_indices <- which(col_data > umi_cutoff)
    col_features <- tibble(cell_bc = rep(col_names[col], length(feature_indices)),
                           feature_call = names(feature_indices),
                           num_umis = col_data[feature_indices])
    features[[col]] <- col_features
  }
  return(bind_rows(features))
}


