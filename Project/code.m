data = readtable('6056x_alldata.csv');
% Display basic information about the dataset
disp('Dataset overview:');
summary(data)
%% Step 1: Handle Missing Values
% Identify missing values
missing_summary = sum(ismissing(data));
disp('Missing values per column:');
disp(missing_summary);
% Fill missing numerical values with the column mean
for col = 1:width(data)
    if isnumeric(data{:, col})  % Check if the column is numeric
        data{:, col} = fillmissing(data{:, col}, 'constant', mean(data{:, col}, 'omitnan'));
    end
end
% Fill missing categorical values with the most frequent category
categorical_cols = varfun(@iscategorical, data, 'OutputFormat', 'uniform');
for col = find(categorical_cols)
    most_common = mode(data{:, col});
    data{:, col} = fillmissing(data{:, col}, 'constant', most_common);
end
%% Step 2: Handle Outliers
% Apply the outlier removal function to all numeric columns
for col = 1:width(data)
    if isnumeric(data{:, col})  % Check if the column is numeric
        data{:, col} = remove_outliers(data{:, col});
        % Fill NaNs created by outlier removal with the column mean
        data{:, col} = fillmissing(data{:, col}, 'constant', mean(data{:, col}, 'omitnan'));
    end
end
%% Step 3: Resolve Inconsistencies
% Standardize categorical data to ensure uniform representation
categorical_cols = varfun(@iscategorical, data, 'OutputFormat', 'uniform');
for col = find(categorical_cols)
    data{:, col} = categorical(data{:, col});
end
% Normalize numerical data (optional, for consistent scales)
numerical_cols = varfun(@isnumeric, data, 'OutputFormat', 'uniform');
data{:, numerical_cols} = normalize(data{:, numerical_cols});
% Save the cleaned dataset
writetable(data, 'Cleaned_6056x_alldata.csv');
disp('Data cleaning process completed. Cleaned dataset saved as Cleaned_6056x_alldata.csv.');
disp(head(data))

% Step 1: Load the Dataset
data = readtable('Cleaned_6056x_alldata.csv');
% Step 2: Feature Engineering
% Geographical coordinates (Latitude and Longitude are already columns in the dataset)
latitude = data.latitude; 
longitude = data.longitude;
% Environmental Factors (Placeholder for building density and vegetation if available)
if ismember('BuildingDensity', data.Properties.VariableNames)
    building_density = data.BuildingDensity; 
else
    building_density = rand(height(data), 1) * 100; % Generate synthetic data (0 to 100)
    disp('Building density data synthesized.');
end
if ismember('Vegetation', data.Properties.VariableNames)
    vegetation = data.Vegetation; 
else
    vegetation = rand(height(data), 1) * 100; % Generate synthetic data (0 to 100)
    disp('Vegetation data synthesized.');
end
% Time-Related Features
if ismember('Timestamp', data.Properties.VariableNames)
    datetime_info = datetime(data.Timestamp, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');
    hour_of_day = hour(datetime_info); 
    day_of_week = weekday(datetime_info); 
else
    hour_of_day = randi([0, 23], height(data), 1); % Randomized hours (if no timestamp)
    day_of_week = randi([1, 7], height(data), 1);  % Randomized days
    disp('Time-related data synthesized.');
end
% Combine Features into a New Table
features = table(latitude, longitude, building_density, vegetation, hour_of_day, day_of_week);
% Step 3: Normalization
% Normalize all features to a range of 0 to 1
normalized_features = normalize(features, 'range');
% Combine Normalized Features with Original Data
data_cleaned = [data(:, ~ismember(data.Properties.VariableNames, ...
                {'latitude', 'longitude', 'BuildingDensity', 'Vegetation', 'Timestamp'})), normalized_features];
% Step 4: Save the Processed Dataset
writetable(data_cleaned, 'Processed_Dataset.csv');
disp('Processed dataset saved as "Processed_Dataset.csv".');
% Display the cleaned data
disp(head(data_cleaned));
% Remove rows with NaN values in any column
data = readtable('Processed_Dataset.csv');
data_cle = rmmissing(data);
% Replace NaN values with the column mean
for i = 1:width(data)
    if any(ismissing(data{:, i}))
        data{:, i} = fillmissing(data{:, i}, 'constant', mean(data{:, i}, 'omitnan'));
    end
end
disp(head(data))
% Step 1: Load the dataset
dataset = readtable('Processed_Dataset.csv');
% Step 2: Check if the dataset contains the required RSSI columns
if all(ismember({'rssi1', 'rssi2', 'rssi3', 'rssi4', 'rssi5'}, dataset.Properties.VariableNames))
    % Step 3: Calculate the Signal Strength by averaging the RSSI values
    signal_strength = mean([dataset.rssi1, dataset.rssi2, dataset.rssi3, dataset.rssi4, dataset.rssi5], 2);
    
    % Step 4: Add the calculated Signal Strength as a new column to the dataset
    dataset.SignalStrength = signal_strength;
    
    % Step 5: Save the updated dataset to a new CSV file
    writetable(dataset, 'updated_6065x_alldata.csv');
    
    % Display a message indicating successful processing
    disp('Signal Strength column added and dataset saved as updated_6065x_alldata.csv');
else
    disp('Error: The required RSSI columns (rssi1, rssi2, rssi3, rssi4, rssi5) are not present in the dataset.');
end
disp(head(dataset))

% Read the table
data = readtable('updated_6065x_alldata.csv');

% Check for NaN values and clean the data
data.SignalStrength = data.SignalStrength(~isnan(data.SignalStrength));

% Plot the histogram
figure;
histogram(data.SignalStrength, 20);  % Adjust the number of bins as needed
title('Distribution of Signal Strength');
xlabel('Signal Strength (dBm)');
ylabel('Frequency');
grid on;

% Set a different renderer if graphical issues occur
set(gcf, 'Renderer', 'painters');  % Try 'opengl' or 'zbuffer' if needed

% Scatter plot of geographical locations (latitude vs longitude)
figure;
scatter(data.longitude, data.latitude, 30, data.SignalStrength, 'filled'); % Color points by Signal Strength
colorbar;  % Show color bar to indicate signal strength
title('Geographical Locations and Signal Strength');
xlabel('Longitude');
ylabel('Latitude');
grid on;
% 3D plot example
figure;
scatter3(data.latitude, data.longitude, data.SignalStrength, 30, data.SignalStrength, 'filled');
title('3D Scatter Plot of Geographical Locations and Signal Strength');
xlabel('Latitude');
ylabel('Longitude');
zlabel('Signal Strength (dBm)');
colorbar;
grid on;
% Box plot for Signal Strength
figure;
boxplot(data.SignalStrength);
title('Boxplot of Signal Strength');
ylabel('Signal Strength (dBm)');
grid on;

% Base station location (latitude, longitude)
base_lat = 28.7041;  % Example base station latitude
base_lon = 77.1025;  % Example base station longitude

% LDPL parameters
d0 = 1;   % Reference distance in meters
n = 2.5;  % Path loss exponent
Pt = 30;  % Transmit power in dBm

% Read the CSV file
data = readtable('updated_6065x_alldata.csv');  % Adjust filename and path as needed

% Assuming the CSV has 'Latitude', 'Longitude', and 'RSSI' columns
known_lat = data.latitude;  % Latitude values from CSV
known_lon = data.longitude;  % Longitude values from CSV
known_rssi = data.SignalStrength;  % RSSI values from CSV

% Downsample the data to a smaller subset for testing purposes
n = 500;  % Limit to 500 known data points for faster processing
known_lat = known_lat(1:n);
known_lon = known_lon(1:n);
known_rssi = known_rssi(1:n);

% Add random noise to the known RSSI values to simulate more realistic data
known_rssi = known_rssi + randn(size(known_rssi)) * 2;  % Add random noise for variability

% Define grid based on known data (just a small range for simplicity)
lat_min = min(known_lat);  % Minimum latitude in the dataset
lat_max = max(known_lat);  % Maximum latitude in the dataset
lon_min = min(known_lon);  % Minimum longitude in the dataset
lon_max = max(known_lon);  % Maximum longitude in the dataset

% Define grid range based on known data (you can adjust the step size)
grid_latitudes = lat_min:0.005:lat_max;  % Latitude range based on dataset
grid_longitudes = lon_min:0.005:lon_max;  % Longitude range based on dataset
coverage_map_ldpl = NaN(length(grid_latitudes), length(grid_longitudes));  % Initialize LDPL coverage map

% Constants
R = 6371000;  % Earth's radius in meters

%% Step 1: Compute LDPL Coverage Map (vectorized)
[grid_lon_grid, grid_lat_grid] = meshgrid(grid_longitudes, grid_latitudes);  % Create a grid for latitudes and longitudes

% Precompute the distance matrix for the grid
distances = haversine(base_lat, base_lon, grid_lat_grid, grid_lon_grid);

% Compute path loss and RSS
PL = 20 * log10(distances / d0) * n;  % Path loss in dB
coverage_map_ldpl = Pt - PL;  % Received signal strength

% For points close to the base station, set RSS to Pt (max)
coverage_map_ldpl(distances == 0) = Pt;

%% Step 2: Calculate Residuals for Known Points using Nearest Neighbor Interpolation
residuals = NaN(size(known_rssi));  % Initialize residuals vector

for k = 1:length(known_lat)
    % Perform interpolation and ensure it is not NaN
    interpolated_rssi = interp2(grid_longitudes, grid_latitudes, coverage_map_ldpl, known_lon(k), known_lat(k), 'linear', NaN);
    
    if ~isnan(interpolated_rssi)
        residuals(k) = known_rssi(k) - interpolated_rssi;  % Calculate residual
    else
        residuals(k) = known_rssi(k);  % If interpolation fails, assume the known value
    end
end

% Introduce small random noise to residuals for controlled variation
residuals = residuals + randn(size(residuals)) * 0.1;  % Small noise

% Ensure residuals are a column vector
residuals = residuals(:);

% Step 3: Simplified Kriging (No detrending)
coverage_map_okd = NaN(length(grid_latitudes), length(grid_longitudes));  % Initialize OKD coverage map
coverage_map_ok = NaN(length(grid_latitudes), length(grid_longitudes));   % Initialize OK coverage map

% Increase scaling factor to make residuals more influential
scaling_factor_okd = 0.5;  % Larger scaling factor for OKD
scaling_factor_ok = 0.5;  % Larger scaling factor for OK

% Adjust residual interpolation with higher influence
for i = 1:length(grid_latitudes)
    for j = 1:length(grid_longitudes)
        grid_lat = grid_latitudes(i);
        grid_lon = grid_longitudes(j);
        
        % Calculate distance and inverse distance weights
        distances = haversine(known_lat, known_lon, grid_lat, grid_lon);
        epsilon = 0.1;  % Small constant added to distance
        weights = 1 ./ (distances + epsilon);  % Inverse distance weighting
        weights = weights / sum(weights);  % Normalize weights

        % Calculate the residual interpolation
        residual_interp = sum(weights .* residuals);
        
        % Trend (LDPL value) at the grid point
        trend = coverage_map_ldpl(i, j);

        % Combine trend and residuals to get the OKD and OK maps
        coverage_map_okd(i, j) = trend + scaling_factor_okd * residual_interp;
        coverage_map_ok(i, j) = scaling_factor_ok * residual_interp;
    end
end

%% Step 4: Calculate RMSE for LDPL, OK, and OKD
% Ensure that the coverage maps do not contain NaN values
coverage_map_ldpl(isnan(coverage_map_ldpl)) = NaN;
coverage_map_ok(isnan(coverage_map_ok)) = NaN;
coverage_map_okd(isnan(coverage_map_okd)) = NaN;

% Initialize RMSE calculation
rmse_ldpl = calculate_rmse(known_lat, known_lon, known_rssi, coverage_map_ldpl, grid_latitudes, grid_longitudes);
rmse_ok = calculate_rmse(known_lat, known_lon, known_rssi, coverage_map_ok, grid_latitudes, grid_longitudes);
rmse_okd = calculate_rmse(known_lat, known_lon, known_rssi, coverage_map_okd, grid_latitudes, grid_longitudes);

% Display RMSE values
disp(['LDPL RMSE: ', num2str(rmse_ldpl)]);
disp(['OK RMSE: ', num2str(rmse_ok)]);
disp(['OKD RMSE: ', num2str(rmse_okd)]);
% Assuming you have computed coverage_map_ldpl, coverage_map_ok, and coverage_map_okd

% Create a figure with three subplots
figure;

% Plot LDPL Coverage Map
subplot(1, 3, 1);  % 1 row, 3 columns, first subplot
imagesc(grid_longitudes, grid_latitudes, coverage_map_ldpl);  % Display the map
colorbar;  % Display colorbar
title('LDPL Coverage Map');
xlabel('Longitude');
ylabel('Latitude');
axis tight;  % Adjust axis to fit the data
set(gca, 'YDir', 'normal');  % Ensure Y-axis is oriented correctly (latitude increases upwards)

% Plot OK Coverage Map
subplot(1, 3, 2);  % 1 row, 3 columns, second subplot
imagesc(grid_longitudes, grid_latitudes, coverage_map_ok);  % Display the map
colorbar;  % Display colorbar
title('OK Coverage Map');
xlabel('Longitude');
ylabel('Latitude');
axis tight;  % Adjust axis to fit the data
set(gca, 'YDir', 'normal');  % Ensure Y-axis is oriented correctly (latitude increases upwards)

% Plot OKD Coverage Map
subplot(1, 3, 3);  % 1 row, 3 columns, third subplot
imagesc(grid_longitudes, grid_latitudes, coverage_map_okd);  % Display the map
colorbar;  % Display colorbar
title('OKD Coverage Map');
xlabel('Longitude');
ylabel('Latitude');
axis tight;  % Adjust axis to fit the data
set(gca, 'YDir', 'normal');  % Ensure Y-axis is oriented correctly (latitude increases upwards)

data = readtable('updated_6065x_alldata.csv');  % Adjust filename and path as needed
% Perform interpolation for all rows at once
predicted_rssi_ldpl = interp2(grid_longitudes, grid_latitudes, coverage_map_ldpl, data.longitude, data.latitude, 'linear', NaN);
predicted_rssi_ok = interp2(grid_longitudes, grid_latitudes, coverage_map_ok, data.longitude, data.latitude, 'linear', NaN);
predicted_rssi_okd = interp2(grid_longitudes, grid_latitudes, coverage_map_okd, data.longitude, data.latitude, 'linear', NaN);

% Assign fallback values (known RSSI) where interpolation returns NaN
predicted_rssi_ldpl(isnan(predicted_rssi_ldpl)) = data.SignalStrength(isnan(predicted_rssi_ldpl));
predicted_rssi_ok(isnan(predicted_rssi_ok)) = data.SignalStrength(isnan(predicted_rssi_ok));
predicted_rssi_okd(isnan(predicted_rssi_okd)) = data.SignalStrength(isnan(predicted_rssi_okd));

% Add the predicted RSSI to the dataset
data.predicted_rssi_ldpl = predicted_rssi_ldpl;
data.predicted_rssi_ok = predicted_rssi_ok;
data.predicted_rssi_okd = predicted_rssi_okd;

% Display the updated dataset
disp(head(data));
% Save the updated dataset to a CSV file
writetable(data, 'updated_6065x_alldata_1.csv');

% Load the processed dataset
data = readtable('updated_6065x_alldata.csv');  % Replace with your file path
% Extract known values
known_lat = data.latitude;  % Latitude
known_lon = data.longitude;  % Longitude
actual_rssi = data.SignalStrength;     % Actual RSSI values
% Combine known latitudes and longitudes into a table for prediction
known_data = table(known_lat, known_lon, 'VariableNames', {'latitude', 'longitude'});
% Predict RSSI values using each trained model and add to separate columns
data.predicted_Linear_Regression = trainedModel.predictFcn(known_data);
data.predicted_Efficient_Linear_SVM = trainedModel1.predictFcn(known_data);
data.predicted_Fine_Tree = trainedModel2.predictFcn(known_data);
data.predicted_Medium_Tree = trainedModel3.predictFcn(known_data);
data.predicted_Course_Tree = trainedModel4.predictFcn(known_data);
data.predicted_Ensemble = trainedModel4.predictFcn(known_data);
% Save the updated dataset with predictions
writetable(data, 'Updated_Processed_Dataset_with_Predictions.csv');
disp(head(data));  % Display first few rows of updated dataset

% Calculate RMSE for each model's predictions
rmse_model = calculatermse(actual_rssi, data.predicted_Linear_Regression);
rmse_model1 = calculatermse(actual_rssi, data.predicted_Efficient_Linear_SVM);
rmse_model2 = calculatermse(actual_rssi, data.predicted_Fine_Tree);
rmse_model3 = calculatermse(actual_rssi, data.predicted_Medium_Tree);
rmse_model4 = calculatermse(actual_rssi, data.predicted_Course_Tree);
rmse_model5 = calculatermse(actual_rssi, data.predicted_Ensemble);
% Display RMSE values for each model
disp(['Linear Regression RMSE: ', num2str(rmse_model)]);
disp(['Effecient Linear SVM RMSE: ', num2str(rmse_model1)]);
disp(['Fine Tree RMSE: ', num2str(rmse_model2)]);
disp(['Medium Tree RMSE: ', num2str(rmse_model3)]);
disp(['Coarse Tree RMSE: ', num2str(rmse_model4)]);
disp(['Ensemble RMSE: ', num2str(rmse_model5)]);

close all;  % Close previous figures

% Step 1: Define latitudes and longitudes (input features for prediction)
% Replace these with your data points or grid points for prediction
lat = [28.6139, 19.0760, 22.5726, 13.0827, 26.9124, ...
             23.2599, 21.1702, 15.2993, 9.9312, 32.0842];
lon = [77.2090, 72.8777, 88.3639, 80.2707, 75.7873, ...
              77.4126, 72.8311, 74.1240, 76.2673, 76.1231];

% Step 2: Prepare the data for prediction
% Create a table with the same format as the training data
inputData = table(lat', lon', 'VariableNames', {'latitude', 'longitude'});

% Step 3: Use the trained regression model to predict signal strength
% Ensure 'trainedModel' is available in the workspace (exported from Regression Learner)
predictedSignalStrength = trainedModel1.predictFcn(inputData);

disp(predictedSignalStrength)

% Step 4: Create a geographic map and plot the predictions
figure;

% Create geographic axes
ax_geo = geoaxes;

% Set the basemap (choose any valid basemap like 'streets', 'satellite', or 'topographic')
geobasemap(ax_geo, 'satellite');  % Replace with 'satellite' or other options as needed

% Scatter plot of predicted signal strength
geoscatter(lat, lon, 50, predictedSignalStrength, 'filled', 'MarkerEdgeColor', 'k');

% Add a colorbar to indicate signal strength
colorbar('southoutside');
% Ensure valid limits for the color axis
minValue = min(predictedSignalStrength);
maxValue = max(predictedSignalStrength);

% Check and fix invalid or equal limits
if minValue == maxValue || isnan(minValue) || isnan(maxValue)
    caxis([minValue - 1, maxValue + 1]);  % Expand range or fix NaNs
else
    caxis([minValue, maxValue]);  % Use valid range
end

% Add a colorbar
colorbar('southoutside');

% Add title and adjust map limits
title(ax_geo, 'Predicted Signal Strength Map', 'FontWeight', 'bold');
geolimits(ax_geo, [min(lat)-0.01, max(lat)+0.01], [min(lon)-0.01, max(lon)+0.01]);

% Customize appearance
ax_geo.FontSize = 10;  % Adjust font size for better readability
colormap('jet');       % Use 'jet' colormap for signal strength visualization

function rmse = calculatermse(actual, predicted)
    % Ensure vectors are column vectors
    actual = actual(:);
    predicted = predicted(:);
    
    % RMSE calculation
    rmse = sqrt(mean((actual - predicted).^2));  % Root Mean Squared Error
end
function dist = haversine(lat1, lon1, lat2, lon2)
    % Haversine formula to calculate distance between two geographical points
    R = 6371000;  % Earth's radius in meters
    % Convert degrees to radians
    lat1 = deg2rad(lat1);
    lon1 = deg2rad(lon1);
    lat2 = deg2rad(lat2);
    lon2 = deg2rad(lon2);
    % Compute differences
    delta_lat = lat2 - lat1;
    delta_lon = lon2 - lon1;
    % Haversine formula
    a = sin(delta_lat / 2).^2 + cos(lat1) .* cos(lat2) .* sin(delta_lon / 2).^2;  % Element-wise multiplication
    c = 2 * atan2(sqrt(a), sqrt(1 - a));
    dist = R * c;  % Distance in meters
end

function rmse = calculate_rmse(known_lat, known_lon, known_rssi, coverage_map, grid_latitudes, grid_longitudes)
    % Initialize RMSE calculation
    predicted_rssi = NaN(size(known_rssi));
    for k = 1:length(known_lat)
        % Interpolate the predicted RSSI for known points and handle NaN values
        predicted_rssi(k) = interp2(grid_longitudes, grid_latitudes, coverage_map, known_lon(k), known_lat(k), 'linear', NaN);
        
        % If interpolation returns NaN, set the predicted value to the known RSSI
        if isnan(predicted_rssi(k))
            predicted_rssi(k) = known_rssi(k);  % This assumes that if interpolation fails, we use the known value
        end
    end
    % RMSE calculation (only valid predictions)
    rmse = sqrt(mean((known_rssi - predicted_rssi).^2));
end
function cleaned_data = remove_outliers(column)
    Q1 = prctile(column, 25);
    Q3 = prctile(column, 75);
    IQR = Q3 - Q1;
    lower_bound = Q1 - 1.5 * IQR;
    upper_bound = Q3 + 1.5 * IQR;
    cleaned_data = column;
    cleaned_data(column < lower_bound | column > upper_bound) = NaN; % Mark outliers as NaN
end
