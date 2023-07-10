library(easyRT)
library(tidyverse)
library(patchwork)

set.seed(3)

sim <- ddm_data(n=3000, drift = c(0, 1.1), bs = 1, bias = c(0.4, 0.6), ndt = c(0, 0.05), n_traces = 5, ndt_var=0)

sim$data <- mutate(sim$data, condition = fct_recode(condition, "Speed" = "1", "Accuracy" = "2"))
sim$traces <- mutate(sim$traces, condition = fct_recode(condition, "Speed" = "1", "Accuracy" = "2"))
sim$density <- mutate(sim$density, condition = fct_recode(condition, "Speed" = "1", "Accuracy" = "2"))


upper <- ddm_plot_upper(sim$data, breaks = 150, density = sim$density, xlim = c(0, 1), density_linewidth=2) +
    # geom_segment(aes(x = 0, y = 0, xend = min(sim$density$x), yend = 0), color = "#4063D8", linewidth=0.8) +
    scale_color_manual(values = c("#4063D8", "#389826")) +
    scale_fill_manual(values = c("#4063D8", "#389826")) +
  theme(axis.title.y = element_blank(),
        panel.grid.major = element_blank())
text <- patchwork::wrap_elements(grid::rasterGrob(png::readPNG("text.png")))

lower <- ddm_plot_lower(sim$data, breaks = 150, density = sim$density, xlim = c(0, 1), density_linewidth=2) +
            scale_color_manual(values = c("#CB3C33", "#9558B2")) +
            scale_fill_manual(values = c("#CB3C33", "#9558B2")) +
  theme(axis.title.y = element_blank(),
        panel.grid.major = element_blank(),
        axis.title.x = element_blank(),
        axis.text.x = element_blank())

p <- (upper) / text / (lower) + patchwork::plot_layout(heights=c(1.5, 1, 1.5))
p
ggsave("logo.png", p, width=20, height=(1.83*4), dpi=300)

# (ddm_plot_traces(sim$traces, trace_alpha = 1, trace_linewidth = 0.2, xlim = c(0, 1)) +
#      scale_color_manual(values = c("#3949AB", "#FF5722")))

/
