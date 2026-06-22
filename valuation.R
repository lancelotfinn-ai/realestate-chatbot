# valuation.R — placeholder ballpark estimator.
# Proves that Python can hand data to R and get structured data back.
# Later, swap this body for your real hedonic model.

suppressMessages(library(jsonlite))

args  <- commandArgs(trailingOnly = TRUE)
input <- fromJSON(args[1])

bedrooms  <- if (is.null(input$bedrooms))  3 else input$bedrooms
bathrooms <- if (is.null(input$bathrooms)) 2 else input$bathrooms

est <- 180000 + bedrooms * 30000 + bathrooms * 18000

result <- list(
  estimate_low  = round(est * 0.9),
  estimate_high = round(est * 1.1),
  basis = "rough placeholder formula, not a real valuation"
)

cat(toJSON(result, auto_unbox = TRUE))
