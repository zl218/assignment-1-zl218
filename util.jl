using DataFrames
using Dates
using Unitful
using CSV
using TidierFiles
using Distributions
using Random

function load_or_sample(fname, model; overwrite = false, n_chains = 4, samples_per_chain = 2000, sampler = NUTS(), threading = MCMCThreads(), rng = rng)
	idata = try
		@assert !overwrite "Reading from cache disabled by overwrite=true"
		idata = ArviZ.from_netcdf(fname)
		@info "Loaded cached samples from $fname"
		return idata
	catch
		chains = sample(
			rng,
			model,
			sampler,
			threading,
			Int(ceil(samples_per_chain * n_chains)),
			n_chains,
			verbose = false,
		)
		idata = ArviZ.from_mcmcchains(chains)
		ArviZ.to_netcdf(idata, fname)
		@info "Sampled and cached samples to $fname"
		return idata
	end
end

"""
    parse_station_parts(station_number::Int, id, name, state, lat_str, lon_str, elevation, years_of_data::Int=0)

Parse station header with 6 parts: ID, NAME, STATE, LAT, LON, ELEVATION.

# Arguments
- `station_number::Int`: Sequential station identifier
- `id`: NOAA station ID string
- `name`: Station name
- `state`: State abbreviation
- `lat_str`: Latitude as string
- `lon_str`: Longitude as string
- `elevation`: Station elevation (not used in calculations)
- `years_of_data::Int=0`: Number of years of data available

# Returns
Named tuple with station metadata including stnid, noaa_id, name, state, latitude, longitude, years_of_data.
"""
function parse_station_parts(
    station_number::Int, id, name, state, lat_str, lon_str, _elevation, years_of_data::Int=0
)
    return (
        stnid=station_number,
        noaa_id=String(strip(id)),
        name=String(strip(name)),
        state=String(strip(state)),
        latitude=parse(Float64, strip(lat_str)),
        longitude=parse(Float64, strip(lon_str)),
        years_of_data=years_of_data,
    )
end

"""
    parse_station_parts(station_number::Int, id, name, city, state, lat_str, lon_str, elevation, years_of_data::Int=0)

Parse station header with 7 parts: ID, NAME, CITY, STATE, LAT, LON, ELEVATION.
Combines name and city for full station name.

# Arguments
- `station_number::Int`: Sequential station identifier
- `id`: NOAA station ID string
- `name`: Station name
- `city`: City name
- `state`: State abbreviation
- `lat_str`: Latitude as string
- `lon_str`: Longitude as string
- `elevation`: Station elevation (not used in calculations)
- `years_of_data::Int=0`: Number of years of data available

# Returns
Named tuple with station metadata including stnid, noaa_id, name (combined with city), state, latitude, longitude, years_of_data.
"""
function parse_station_parts(
    station_number::Int,
    id,
    name,
    city,
    state,
    lat_str,
    lon_str,
    _elevation,
    years_of_data::Int=0,
)
    # Combine name and city for the full name
    full_name = "$(strip(name)), $(strip(city))"
    return (
        stnid=station_number,
        noaa_id=String(strip(id)),
        name=String(full_name),
        state=String(strip(state)),
        latitude=parse(Float64, strip(lat_str)),
        longitude=parse(Float64, strip(lon_str)),
        years_of_data=years_of_data,
    )
end

"""
    parse_station_header(header_line::AbstractString, station_number::Int, years_of_data::Int=0)

Parse a station header line using method overloading for different formats.
Supports both 6-part and 7-part station headers through method dispatch.

# Arguments
- `header_line::AbstractString`: Comma-separated header line from NOAA data file
- `station_number::Int`: Sequential station identifier
- `years_of_data::Int=0`: Number of years of data available

# Returns
Named tuple with parsed station information.

# Example
```julia
header = "12345, Station Name, TX, 29.76, -95.37, 10"
station = parse_station_header(header, 1, 25)
```
"""
function parse_station_header(
    header_line::AbstractString, station_number::Int, years_of_data::Int=0
)
    # Split by commas and strip whitespace
    parts = [strip(p) for p in split(header_line, ",")]

    # Use splatting with method overloading
    return parse_station_parts(station_number, parts..., years_of_data)
end

"""
    parse_rainfall_data(data_lines::Vector{<:AbstractString}, rainfall_unit, stnid::Int)

Parse rainfall data lines and return DataFrame with complete time series.
Fills missing years with missing values to ensure continuous temporal coverage.

# Arguments
- `data_lines::Vector{<:AbstractString}`: Vector of data lines with "mm/dd/yyyy value" format
- `rainfall_unit`: Unitful unit for rainfall measurements (typically u"inch")
- `stnid::Int`: Station identifier

# Returns
DataFrame with columns: stnid, date, year, rainfall (with units)

# Details
Creates complete time series from minimum to maximum year found in data,
filling missing years with `missing` values and placeholder dates (January 1st).
"""
function parse_rainfall_data(
    data_lines::Vector{<:AbstractString}, rainfall_unit, stnid::Int
)
    dates = Date[]
    years = Int[]
    rainfall = Float64[]
    stnids = Int[]

    # Parse available data
    for line in data_lines
        line = strip(line)
        if !isempty(line)
            parts = split(line)
            if length(parts) >= 2
                date_str = parts[1]
                rain_str = parts[2]

                date = Date(date_str, "mm/dd/yyyy")
                rain_val = parse(Float64, rain_str)

                push!(dates, date)
                push!(years, year(date))
                push!(rainfall, rain_val)
                push!(stnids, stnid)
            end
        end
    end

    # If no data, return empty DataFrame
    if isempty(years)
        return DataFrame(;
            stnid=Int[], date=Date[], year=Int[], rainfall=typeof(1.0 * rainfall_unit)[]
        )
    end

    # Create complete year sequence and fill missing years
    min_year, max_year = extrema(years)
    complete_years = collect(min_year:max_year)

    # Create vectors for complete time series
    complete_stnids = Int[]
    complete_dates = Date[]
    complete_years_vec = Int[]
    complete_rainfall = Union{Float64,Missing}[]

    for yr in complete_years
        push!(complete_stnids, stnid)
        push!(complete_years_vec, yr)

        # Find if this year has data
        year_idx = findfirst(==(yr), years)
        if year_idx !== nothing
            push!(complete_dates, dates[year_idx])
            push!(complete_rainfall, rainfall[year_idx])
        else
            # Missing year - use January 1st as placeholder date
            push!(complete_dates, Date(yr, 1, 1))
            push!(complete_rainfall, missing)
        end
    end

    return DataFrame(;
        stnid=complete_stnids,
        date=complete_dates,
        year=complete_years_vec,
        rainfall=complete_rainfall .* rainfall_unit,
    )
end

"""
    read_noaa_data(filename::String)

Parse NOAA precipitation data file and return structured station and rainfall data.

# Arguments
- `filename::String`: Path to NOAA precipitation data file

# Returns
Tuple of two DataFrames:
- `stations`: Station metadata with columns (stnid, noaa_id, name, state, latitude, longitude, years_of_data)
- `rainfall_data`: Rainfall time series with columns (stnid, date, year, rainfall with units)

# Format
Expects NOAA format with:
- Line 1: Header with units information
- Subsequent blocks: Station header followed by date/rainfall pairs, separated by blank lines

# Example
```julia
stations, rainfall = read_noaa_data("precip_data.txt")
filter(row -> row.stnid == 1, stations)  # Get first station info
filter(row -> row.stnid == 1, rainfall)  # Get first station rainfall data
```
"""
function read_noaa_data(filename::String)
    # Read the data file
    txt = read(filename, String)
    lines = split(txt, '\n')

    # (1) Extract header info and set units  # <1>
    _header = lines[1]  # <2>
    rainfall_unit = u"inch"  # <3>

    remaining_content = join(lines[2:end], '\n')

    # (2) Split by blank lines which separate each gauge
    station_blocks = filter(!isempty, split(remaining_content, r"\n\s*\n"))

    stations = []
    all_rainfall_data = DataFrame[]

    # (3) For each gauge
    for (i, block) in enumerate(station_blocks)
        block_lines = split(strip(block), '\n')

        if !isempty(block_lines)
            # (a) Pull out the header row into station information
            header_line = block_lines[1]

            # (b) Parse the rainfall data with station ID
            data_lines = block_lines[2:end]
            rainfall_df = parse_rainfall_data(data_lines, rainfall_unit, i)
            years_count = sum(.!ismissing.(rainfall_df.rainfall))  # Count only non-missing values

            # Create station with years_of_data
            station = parse_station_header(header_line, i, years_count)
            push!(stations, station)
            push!(all_rainfall_data, rainfall_df)
        end
    end

    # Convert stations vector to DataFrame
    stations_df = DataFrame(stations)

    # Combine all rainfall data into single DataFrame
    rainfall_data_df = vcat(all_rainfall_data...)

    return stations_df, rainfall_data_df
end

"""
    test_read_noaa_data()

Simple test function to verify NOAA data parsing functionality.
Creates minimal test data and validates parsing results.

# Returns
`true` if all tests pass, `false` otherwise.
"""
function test_read_noaa_data()
    # Create minimal test data
    test_content = """1-d, Annual Maximum, WaterYear=1 (January - December), Units in Inches
60-0011, CLEAR CK AT BAY AREA BLVD               , TX,  29.4977,  -95.1599, 2
06/11/1987    6.31
09/02/1988    5.46

60-0019, TURKEY CK AT FM 1959                    , TX,  29.5845,  -95.1869, 28
06/11/1987    3.99
09/02/1988    3.71
"""

    # Write test file
    test_file = "test_noaa.txt"
    write(test_file, test_content)

    try
        # Test the function
        stations, rainfall_data = read_noaa_data(test_file)

        # Basic tests
        @assert length(stations) == 2 "Expected 2 stations, got $(length(stations))"
        @assert length(rainfall_data) == 2 "Expected 2 rainfall datasets, got $(length(rainfall_data))"
        @assert stations[1].stnid == "stn_1" "Expected stn_1, got $(stations[1].stnid)"
        @assert stations[1].noaa_id == "60-0011" "Expected 60-0011, got $(stations[1].noaa_id)"
        @assert nrow(rainfall_data["stn_1"]) == 2 "Expected 2 rainfall records for stn_1"

        println("✓ All tests passed!")
        return true
    catch e
        println("✗ Test failed: $e")
        return false
    finally
        # Clean up test file
        if isfile(test_file)
            rm(test_file)
        end
    end
end

"""
    calc_distance(lon1, lat1, lon2, lat2)

Calculate great-circle distance between two points using the Haversine formula.

# Arguments
- `lon1`, `lat1`: Longitude and latitude of first point (degrees)
- `lon2`, `lat2`: Longitude and latitude of second point (degrees)

# Returns
Distance with units (kilometers)

# Example
```julia
# Distance between Houston and Dallas
dist = calc_distance(-95.37, 29.76, -96.80, 32.78)
```
"""
function calc_distance(lon1, lat1, lon2, lat2)  # <1>
    R = 6378.0u"km"  # <2>

    # Convert degrees to radians
    φ1 = deg2rad(lat1)  # <3>
    φ2 = deg2rad(lat2)  # <4>
    Δφ = deg2rad(lat2 - lat1)  # <5>
    Δλ = deg2rad(lon2 - lon1)  # <6>

    # Haversine formula
    a = sin(Δφ / 2)^2 + cos(φ1) * cos(φ2) * sin(Δλ / 2)^2  # <7>
    c = 2 * atan(sqrt(a), sqrt(1 - a))  # <8>

    return R * c  # <9>
end

"""
    weibull_plotting_positions(data)

Calculate Weibull plotting positions for empirical probability analysis.
Used for creating return period plots and comparing with theoretical distributions.

# Arguments
- `data`: Vector of data values (may contain missing values)

# Returns
Tuple of:
- `clean_data`: Sorted data with missing values removed
- `empirical_return_periods`: Corresponding empirical return periods

# Formula
Plotting position: p = i/(n+1) where i is rank, n is sample size
Return period: T = 1/(1-p)

# Example
```julia
data = [1.2, 2.1, missing, 3.4, 1.8]
values, periods = weibull_plotting_positions(data)
```
"""
function weibull_plotting_positions(data)
    clean_data = sort(collect(skipmissing(data)))  # <1>
    n = length(clean_data)
    plotting_positions = [i / (n + 1) for i in 1:n]  # <2>
    empirical_return_periods = [1 / (1 - p) for p in plotting_positions]  # <3>
    return clean_data, empirical_return_periods
end

"""
    find_nearest_stations(target_station, all_stations, n_nearest::Int=3)

Find the n nearest stations to a target station based on great-circle distance.

# Arguments
- `target_station`: Station row with latitude and longitude
- `all_stations`: DataFrame of all available stations
- `n_nearest::Int=3`: Number of nearest stations to return

# Returns
DataFrame containing the n nearest stations (excluding the target station itself).

# Example
```julia
my_station = stations[5, :]  # Select station 5
nearest = find_nearest_stations(my_station, stations, 3)
```
"""
function find_nearest_stations(target_station, all_stations, n_nearest::Int=3)
    # Calculate distances to all other stations
    distances = []
    station_indices = []

    for (i, station) in enumerate(eachrow(all_stations))
        dist = calc_distance(
            target_station.longitude,
            target_station.latitude,
            station.longitude,
            station.latitude,
        )
        push!(distances, dist)
        push!(station_indices, i)
    end

    # Find indices of n nearest stations
    sorted_indices = sortperm(distances)
    nearest_indices = station_indices[sorted_indices[1:min(
        n_nearest, length(sorted_indices)
    )]]

    return all_stations[nearest_indices, :]
end


"""
    add_return_level_curve!(ax, dist, rts; kwargs...)

Add a return level curve to an axis based on a distribution and return periods.
Computes return levels from the distribution and passes kwargs to lines!.
"""
function add_return_level_curve!(ax, dist, rts; kwargs...)
    return_levels = [quantile(dist, 1 - 1/T) for T in rts]
    lines!(ax, rts, return_levels; kwargs...)
    return ax
end

"""
    ReturnLevelPrior

A struct representing a prior belief about a specific return level.

# Fields
- `quantile::Float64`: The quantile corresponding to the return period (1 - 1/T)
- `distribution::InverseGamma`: The prior distribution for the return level

# Constructor
    ReturnLevelPrior(return_period, mean, stdev)

Creates a return level prior from intuitive parameters.

# Arguments
- `return_period`: The return period in years (e.g., 10 for 10-year event)
- `mean`: Expected return level value
- `stdev`: Standard deviation of return level belief

# Example
```julia
# 10-year event: expect ~8 inches ± 4 inches
prior = ReturnLevelPrior(10, 8.0, 4.0)
```
"""
struct ReturnLevelPrior
    quantile::Float64
    distribution::InverseGamma

    function ReturnLevelPrior(return_period, mean, stdev)
        quantile = 1.0 - 1.0 / return_period

        # Convert mean/stdev to InverseGamma parameters
        variance = stdev^2
        α = 2 + mean^2 / variance
        β = (α - 1) * mean
        distribution = InverseGamma(α, β)

        new(quantile, distribution)
    end
end

"""
    posterior_mean_curve!(ax, dists::Vector{<:Distribution}, rts; kwargs...)

Plot the posterior mean return level curve for a vector of distributions.
"""
function posterior_mean_curve!(ax, dists::Vector{<:Distribution}, rts; kwargs...)
    mean_return_levels = [mean([quantile(dist, 1 - 1/T) for dist in dists]) for T in rts]
    lines!(ax, rts, mean_return_levels; kwargs...)
    return ax
end

"""
    posterior_bands!(ax, dists::Vector{<:Distribution}, rts; ci=0.90, kwargs...)

Plot credible interval bands for return levels from a vector of distributions.
"""
function posterior_bands!(ax, dists::Vector{<:Distribution}, rts; ci=0.90, kwargs...)
    α = (1 - ci) / 2
    q_low, q_high = α, 1 - α

    return_level_quantiles = map(rts) do T
        return_levels = [quantile(dist, 1 - 1/T) for dist in dists]
        (quantile(return_levels, q_low), quantile(return_levels, q_high))
    end

    lower_bound = [q[1] for q in return_level_quantiles]
    upper_bound = [q[2] for q in return_level_quantiles]

    band!(ax, rts, lower_bound, upper_bound; kwargs...)
    return ax
end

"""
    traceplot!(ax, idata, param_name; kwargs...)

Add a simple traceplot for a parameter to an existing axis.

# Arguments
- `ax`: The axis to plot on
- `idata`: ArviZ InferenceData object containing posterior samples
- `param_name`: Symbol or string name of the parameter to plot

# Keyword Arguments
- `color`: Line color (defaults to cycling through colors for each chain)
- `linewidth`: Line width (default: 1.5)
- `alpha`: Line transparency (default: 0.8)

# Example
```julia
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Iteration", ylabel = "μ₀")
traceplot!(ax, idata, :μ₀)
```
"""
function traceplot!(ax, idata, param_name; linewidth = 1.5, alpha = 0.8, kwargs...)
    param_data = Array(idata.posterior[param_name])
    n_chains = size(param_data, 2)

    # Plot each chain
    for chain in 1:n_chains
        if ndims(param_data) == 2
            lines!(ax, param_data[:, chain],
                   color = Cycled(chain), linewidth = linewidth, alpha = alpha; kwargs...)
        else
            # Handle scalar parameters (single value across iterations)
            lines!(ax, param_data,
                   color = Cycled(chain), linewidth = linewidth, alpha = alpha; kwargs...)
        end
    end

    # Add horizontal line at mean for reference
    hlines!(ax, [mean(param_data)], color = :black, linestyle = :dash, alpha = 0.5)

    return ax
end
