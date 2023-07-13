library(easyRT)
library(tidyverse)
library(patchwork)

set.seed(3)

# sim <- ddm_data(n=3000, drift = c(0, 1.1), bs = 1, bias = c(0.4, 0.6), ndt = c(0, 0.05), n_traces = 5, ndt_var=0)
sim <- ddm_data(n=3000, drift = c(1.1, 0), bs = 1, bias = c(0.6, 0.4), ndt = c(0.05, 0), n_traces = 5, ndt_var=0)

sim$data <- mutate(sim$data, condition = fct_recode(condition, "Speed" = "1", "Accuracy" = "2"))
sim$traces <- mutate(sim$traces, condition = fct_recode(condition, "Speed" = "1", "Accuracy" = "2"))
sim$density <- mutate(sim$density, condition = fct_recode(condition, "Speed" = "1", "Accuracy" = "2"))


upper <- ddm_plot_upper(sim$data, breaks = 150, density = sim$density, xlim = c(0, 1), density_linewidth=2, hist_alpha = 0.5) +
    # geom_segment(aes(x = 0, y = 0, xend = min(sim$density$x), yend = 0), color = "#4063D8", linewidth=0.8) +
    scale_color_manual(values = rev(c("#4063D8", "#389826"))) +
    scale_fill_manual(values = rev(c("#4063D8", "#389826"))) +
  theme(axis.title.y = element_blank(),
        panel.grid.major = element_blank())
text <- patchwork::wrap_elements(grid::rasterGrob(png::readPNG("text.png")))

lower <- ddm_plot_lower(sim$data, breaks = 150, density = sim$density, xlim = c(0, 1), density_linewidth=2, hist_alpha = 0.5) +
            scale_color_manual(values = c("#CB3C33", "#9558B2")) +
            scale_fill_manual(values = c("#CB3C33", "#9558B2")) +
  theme(axis.title.y = element_blank(),
        panel.grid.major = element_blank(),
        axis.title.x = element_blank(),
        axis.text.x = element_blank())

p <- (upper) / text / (lower) + patchwork::plot_layout(heights=c(1.5, 1, 1.5))
p

ggsave("logo.png", p, width=20, height=(1.83*4), dpi=300)

p2 <- (upper + theme(plot.background = element_rect(fill = "transparent",colour = NA),
                     panel.background = element_rect(fill = "transparent", colour = NA))) /
  (plot_spacer() + theme(plot.background = element_rect(fill = "transparent",colour = NA),
                        panel.background = element_rect(fill = "transparent", colour = NA))) /
  (lower + theme(plot.background = element_rect(fill = "transparent", colour = NA),
                 panel.background = element_rect(fill = "transparent", colour = NA))) +
  patchwork::plot_layout(heights=c(1.5, -0.175, 1.5)) +
  patchwork::plot_annotation(theme =  theme(plot.background = element_rect(fill = "transparent", colour = NA),
                                            panel.background = element_rect(fill = "transparent", colour = NA),
                                            plot.margin = margin(b = -90)))
p2
ggsave("../src/assets/logo.png", p2, width=12, height=6, dpi=300, bg = "transparent")

