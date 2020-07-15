

# Load package
library(Synth)

# Load Data 
myd <- read.csv("../data/perfectures/basins/all_basins.csv")
myd$Basin <- as.character(myd$Basin)

## pick v by cross-validation
# data setup for training model
dataprep.out <-
  dataprep(
           foo = myd,
           predictors    = c("Irrigated.area..Total", "Industrial.gross.value.added..GVA...Total", "Urban.population", "Service.GVA", "Rural.population", "Livestock.population"),
           dependent     = "Total.water.use",
           unit.variable = "Index",
           time.variable = "Year",
           treatment.identifier = 3,
           controls.identifier = c(0, 1, 2, 4, 5, 6, 7, 8),
           time.predictors.prior = 1965:1994,
           time.optimize.ssr = 1995:2013,
           unit.names.variable = "Basin",
           time.plot = 1965:2013
         )

# fit training model
synth.out <- 
  synth(
        data.prep.obj=dataprep.out,
        Margin.ipop=.005,Sigf.ipop=7,Bound.ipop=6
        )


actual_data <- dataprep.out$Y1plot
synth_data <- dataprep.out$Y0plot %*% synth.out$solution.w 

out_list <- list(
  y1 = actual_data,
  y0 = synth_data
)
