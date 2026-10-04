# How far ahead does today's weather beat the calendar?

Daily high and low temperature and daily rain at twelve US airport stations, 1991 to 2025. Five questions:
- how much of tomorrow's weather is simply the time of year, and how much better than the calendar can a station's own recent weather do;
- how many days ahead that advantage lasts, and whether repeating today's weather ("tomorrow will be like today") is any use;
- how the answer changes between stations and seasons;
- what happens to a calendar that is not kept up to date, in a warming climate;
- the same for rain: will it rain tomorrow?

The questions follow GeeksforGeeks' "Weather Data Analysis" (time series of station weather), answered as a forecasting test with rolling walk-forward fits that only ever see the past. The page with the results and the charts is the build log:

Build log: https://mrrishit909.github.io/projects/weather-forecast-skill/

## Data

NOAA National Centers for Environmental Information, GHCN-Daily (https://www.ncei.noaa.gov/pub/data/ghcn/daily/): the by-station files for New York (Central Park), Chicago (O'Hare), Seattle (Sea-Tac), Phoenix, Miami, Atlanta, Boston (Logan), Minneapolis-St Paul, Los Angeles (LAX), Dallas-Fort Worth, Honolulu and Albuquerque, daily maximum and minimum temperature (TMAX, TMIN) and precipitation (PRCP), 1 January 1991 to 31 December 2025, with the station list `ghcnd-stations.txt`. US government data; the readme asks for citation: M. J. Menne, I. Durre, R. S. Vose, B. E. Gleason and T. G. Houston, 2012, "An overview of the Global Historical Climatology Network-Daily Database", Journal of Atmospheric and Oceanic Technology 29, 897-910, doi:10.1175/JTECH-D-11-00103.1, and the dataset, doi:10.7289/V5D21VHZ. Values with a quality flag are dropped. NOAA rebuilds the files daily, so a later download can differ in recent months and in rare revisions; `results/pull.json` has the checksums of the files used. The files are about 19 MB and are fetched by `download.py`, not committed.

```
./venv/bin/pip install pandas numpy certifi matplotlib      # in a venv
./venv/bin/python download.py    # data/*
./venv/bin/python analyze.py     # results/*   (a few seconds)
./venv/bin/python check.py       # about one minute
./venv/bin/python charts.py
```

## Steps

1. **Download** the twelve station files and the station list, with their checksums.
2. **Audit.** Days observed and values dropped by a quality flag for each element, the wet-day share and the mean temperatures.
3. **Walk-forward fits.** For each test year from 2011 to 2025, the models are fitted on the 20 years before it only. Climatology is the seasonal cycle (a mean and three annual harmonics) plus a linear trend, evaluated at the day being forecast.
4. **Four forecasts of the high and the low, 1 to 14 days ahead.** The calendar (climatology); persistence (today's value again); damped persistence (climatology plus a fitted share of today's departure from it); and "today and last week" (climatology plus two fitted terms: today's departure and the mean departure of the last seven days). All four are scored on the same days by root-mean-square error, and skill is one minus the ratio of mean squared errors to the calendar's, so 0 is no better than the calendar.
5. **Skill by station, horizon and season, and the stale calendar.** The first day on which the damped forecast's skill falls under 5%; the skill by season of the target day; and the bias of a calendar fitted once on 1991 to 2010 (with and without a trend) against the rolling one.
6. **Rain, 1 to 5 days ahead.** The probability that a day has 1 mm or more: the seasonal probability (a mean and two harmonics), persistence (wet if wet today), and a two-state model (a separate seasonal curve after a wet and after a dry day), scored by the Brier score.
7. **Check.** `check.py` rebuilds everything from the station files in plain Python (gzip, csv, math, datetime and loops: no pandas, numpy or scipy), solving every least-squares fit by Gaussian elimination: all 4,227 numbers in the tables, to a relative tolerance of 1e-8 (the two programs sum in a different order).
8. **Charts.**

## Results

- **The data.** 12 stations, 12,784 days each; 99.8% or more of the days have a high and a low, and 4 values are dropped by a quality flag. Each forecast is scored on 5,444 to 5,478 days per station and horizon (fewer for the long horizons near the end of 2025). It rains (1 mm or more) on 7% (Phoenix) to 34% (Seattle) of the days.
- **The calendar is a hard benchmark.** The seasonal cycle and trend alone explain 29% (the high at Los Angeles) to 89% (the low at Phoenix) of the variance of the next day's temperature, so what is left to forecast is the departure from it.
- **Today's weather beats the calendar for about three days.** Averaged over the twelve stations, the damped forecast of the daily high has a skill of 42% tomorrow, 15% in two days, 7% in three, 3% in five, 2% in seven and 1% in fourteen (the daily low: 49%, 20%, 10%, 4%, 2%, 1%). Adding the last week's departure changes almost nothing (42% tomorrow, 8% in three).
- **Repeating today is worse than the calendar from day two.** The skill of "tomorrow will be like today" is 29% for the high tomorrow, then -24%, -50%, -81% and -100% at two, three, seven and fourteen days (once the weather has forgotten today, repeating it doubles the error variance, a skill of -100%). At 19 of the 24 station-and-element combinations it is already worse than the calendar at two days, and at the other 5 at three.
- **Some stations hold on longer.** The damped skill of the high stays above 5% for 1 day at Boston, 2 at New York and Miami, and 9 at Honolulu; for the low, 12 days at Los Angeles and 2 at Boston. Tomorrow's skill runs from 29% (the high at Boston) to 58% (Phoenix), where the weather changes least.
- **The seasons matter less than the station.** Tomorrow's skill for the high runs from 27% to 61% across the station-seasons; at Minneapolis it is 50% in winter and 32% in summer, at Phoenix 53% in summer.
- **An old calendar runs cold.** A seasonal cycle fitted on 1991 to 2010 and left alone under-forecasts 2011 to 2025 by 0.72 °C on average (0.09 to 1.22 over the twelve highs and twelve lows). Adding a linear trend fitted on the same years extrapolates badly: the error runs from -0.65 to 1.50 °C. Refitting on the last 20 years each year keeps it between -0.05 and 0.27 °C. The cost in root-mean-square error of tomorrow's forecast is a median 1% for the stale trend and 2% for the stale seasonal cycle, and at worst 22% and 5%.
- **Rain is hardly forecastable from the station's own history.** The two-state model improves on the seasonal probability of rain tomorrow by 4.5% on average (1.4% to 10.8%; the most at Los Angeles and Seattle), by 0.5% in two days and by nothing after that. "Wet if wet today" is worse than the seasonal probability, with a Brier score of 0.24 against 0.15.

## Not done, and caveats

- These forecasts use only the station's own past. They are not weather forecasts: the forecasts of a weather service use observations from a whole region and physical models, and none of those, nor any published verification, were used or measured here.
- Twelve airport stations are not the country; several are coastal. An airport is not a city, and the Central Park, LAX and Sea-Tac records have their own histories (instrument and site changes were not examined).
- The climatology is one choice (three harmonics, a linear trend, a 20-year window); other windows, more harmonics, or a variance that changes with the season were not tried. Anomalies are assumed to be linear in today's departure.
- Skill is a mean over all days; it is not skill for extreme weather, and the mean squared error ignores what a forecast would be used for.
- "A day with rain" is 1 mm or more; other thresholds, amounts and snow were not examined. The linear-probability seasonal curves are clipped to 0.001 to 0.999.
- Values with a quality flag are dropped, not repaired; days with a missing value are skipped in the scoring (a forecast needs today, the target day and, for the week term, the seven days to today).
- NOAA states no licence beyond asking for citation; nothing from the data is committed, only station summaries.
