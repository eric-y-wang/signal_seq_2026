# sourced from https://github.com/koundy/ggplot_theme_Publication/blob/master/ggplot_theme_Publication-2.R
theme_Publication <- function(base_size=14, base_family="sans") {
  library(grid); library(ggthemes)
  theme_foundation(base_size=base_size, base_family=base_family) +
    theme(
      plot.title = element_text(face="bold", size=rel(1.2), hjust=0.5, margin=margin(0,0,20,0)),
      panel.background = element_rect(fill="white", colour=NA),
      plot.background  = element_rect(fill="white", colour=NA),
      panel.border = element_rect(colour=NA),
      axis.title = element_text(face="bold", size=rel(1), colour="black"),
      axis.title.y = element_text(angle=90, vjust=2, colour="black"),
      axis.title.x = element_text(vjust=-0.2, colour="black"),
      axis.text = element_text(colour="black"),
      axis.line.x = element_line(colour="black"),
      axis.line.y = element_line(colour="black"),
      axis.ticks = element_line(),
      panel.grid.major = element_line(colour="#f0f0f0"),
      panel.grid.minor = element_blank(),
      legend.key = element_rect(fill="white", colour=NA),
      legend.position = "right",
      legend.direction = "vertical",
      legend.box = "vertical",
      legend.key.size= unit(0.5, "cm"),
      legend.title = element_text(face="bold"),
      plot.margin=unit(c(10,5,5,5),"mm"),
      strip.background=element_rect(colour="#f0f0f0", fill="#f0f0f0"),
      strip.text = element_text(face="bold", colour="black")
    )
}


# function to save pheatmap output
save_pheatmap_pdf <- function(x, filename, width=7, height=7) {
  stopifnot(!missing(x))
  stopifnot(!missing(filename))
  pdf(filename, width=width, height=height)
  grid::grid.newpage()
  grid::grid.draw(x$gtable)
  dev.off()
}