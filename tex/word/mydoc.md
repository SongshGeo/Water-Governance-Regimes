# My doc

Here, we constructed the Integrated Water Resources Utilization (IWRU) Index which consists of three dimensions and identified the changes periods of the index over time by change points detection. Each dimension is reflected by an independent indicator after normalization, and water utilization regime were characterized by combination of impacts of each dimension in periods. In addition, the contribution to changes of IWRU index along with each main indicators were decomposed and calculated separately for each regime (i.e. period). 

## Integrated Water Resources Utilization (IWRU) Index

Water resources utilization system is closely related to the social developments in three dimensions below (see *SI Appendix* Methods S4 for details):

Social development is usually accompanied by a change of priority in water use towards social and economic systems because of higher returns:

$$ Dev. \propto P $$

Social development usually lead to more complex structure in configuration of water resources, which is a result of division and cooperation between regions and sectors for developing:

$$ Dev \propto C $$

Further social development can only be achieved by effectively alleviating the water resource stresses generated in the process of development through technological means:

$$ Dev \propto S^{-1} $$


We combine the above three dimensions for an integrated index, remaining their positive or negative relationship with social development: 

$$ Dev. \propto P*C*S^{-1} $$

To effectively represent the three dimensions, we select an appropriate indicator ($I_x$, $x=P$, $C$ or $S$ corresponding to Priority, Configuration and Stress respectively) for each dimension. Then, the above equation is transformed into a natural logarithm to facilitate calculation:

$$ Dev. \propto ln(I_P) + ln(I_C) - ln(I_S) $$

Assuming they have equal weights, the Integrated Water Resources Utilization (IWRU) Index:

$$ IWRU = I'_T + I'_P - I'_S $$

where $I'_x$ is a normalization of log-transformed indicator $I_x$ for a certain dimension:

$$ I'_x = normalize(ln(I_x)) $$

In fact, we have tested different normalization methods and it makes no difference in change points detection (see \textit{SI Appendix} Methods S5. Sensitivity analysis). In this study, finally, we performed min-max normalization as the formulation below:

$$ normalize(X) = (X - X_{min}) / (X_{max} - X_{min}) $$

## Indicator of Stress

We refer to the scarcity-flexibility-variability (SFV) water stress index proposed in Qin et al., 2019 to evaluate water stress ($SFV_i$) as the indicator in a certain region $i$ \cite{qinFlexibilityIntensityGlobal2019}. This metric takes into account management measures (such as the construction of reservoirs) and the impact of changes in the industrial structure of water use on the evaluation of water scarcity (see \textit{SI Appendix} Methods S4 for details). For the whole YRB, indicator of stress $I_S$ is the average stress of all regions' SFV-index: 

$$ I_S = \frac{1}{4} * \sum_{i=1}^4 SFV_{i} $$

Where $SFV_i$ is the SFV-index for region $i$, and $i=1$ to $4$ refers SR, UR, MR, and DR (see \textit{SI Appendix} Methods S1 Definition of study area).

## Indicator of priority

To priority $I_P$, we use Non-Provisioning Shares (NPS) of water use as an indicator. While provisional water use ($WU_{pro}$) includes domestic, irrigated and livestock water uses, the non-provisioning water use ($WU_{non-pro}$) includes industrial and urban services water uses. Then, we can calculate the non-provisioning shares by:

$$ NPS_{i} = \frac{WU_{non-pro, i}}{WU_{pro, i} + WU_{non-pro, i}} $$

Where $i$ refers a certain region, or the whole basin, i.e:

$$ I_P = NPS_{basin} $$

Indicator of configurations

To description of configurations $I_C$, we designed an indicator by imitation of information entropy, called Configuration Entropy Metric (CEM), a metric to measure the degree of evenness of water configuration (see \textit{SI Appendix} Methods S4).
While our indicator $I_C$ should reflect that with the development of society, water resources configuration is more balanced among regions and generally meets the needs of different sectors (means smaller gaps, too), but different regions have a trend of division of labour among various sectors (with larger gaps):

$$ I_C = \frac{CEM_{r}*CEM_{s}}{CEM_{rs}}$$

where $CEN_{r}$ and $CEN_{s}$ are Configuration Entropy Metric in different regions and different sectors. $CEN_{rs}$ is differences between sectors in a certain region to the whole basin (see \textit{SI Appendix} Methods S4). 

## Change points detection

With no assumptions about the distribution of the data, the Pettitt (1979) approach of changing points detection is commonly applied to detect a single change-point in hydrological series with continuous data \cite{pettittNonParametricApproachChangePoint1979}. It tests the $H0$: The variables follow one or more distributions that have the same location parameter (no change), against the alternative: a change point exists. The non-parametric statistic is defined as:

$$ K_t = max|U_{t, T}|$$

Where:

$$ U_{t, T} = \sum_{i=1}^t\sum_{j=t+1}^T sgn(X_i - X_j) $$

The change-point of the series is located at $K_T$, provided that the statistic is significant. We use 0.001 as the threshold  of p-value (see \textit{SI Appendix} Methods S5 for Sensitivity analysis), which means the probability of a statistically significant change-point judgment being valid is more than $99.9\%$. Since this method only can return one significant change point, we repeat it Until all significant change points were detected.

## Contribution decomposition

We have decomposed the amount of variation in each index at different stages in order to observe the contribution of each influencing factor to them. Use Integrated Water Resources Utilization (IWRU) Index as an example, which influenced by three dimensions' normalized indicator: stress ($I'_S$), priority ($I'_P$) and configuration ($I'_C$). We can calculate their differences between two certain years ($y_2$ and $y_1$, $y_2 > y_1$) by:


Then, the contribution of dimension $x$ to IWRU's changes can be referred as:

$$ Contribution_x = \frac{\Delta I'_x}{|\Delta IWRU|} $$

Datasets

In order to calculate IWRU, we need to calculate multiple indicators and sub-indicators. All the datasets used are listed in the \textit{SI Appendix} table S1. A detailed description of the data can be seen in the supplementary materials \textit{SI Appendix} Methods S2.