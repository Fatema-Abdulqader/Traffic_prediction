% MAIN SCRIPT: Hybrid GRU->LSTM with Additive Attention for Traffic Forecasting
% Paper: Advancing Urban Planning with Deep Learning: Intelligent Traffic Flow Prediction and Optimization for Smart Cities
clear; clc; rng(42);

%% ------------------------ Load & Prepare Data ---------------------------
% Your data should be arranged as:
%   data.speed, data.flow, data.occ   : T x S matrices (time x sensors)
%   dataTime (datetime or numeric indices)
%   adjList : 1xS cell, each cell contains a vector of 1-hop neighbor indices N(i)
%
% Replace this loader with your own PeMS ingestion.
[data, dataTime, adjList] = load_pems_example();  % <- dummy example loader

% Choose the variable to forecast (e.g., speed). You can switch to flow/occ.
Xraw = data.speed;   % T x S

% Split indices by time (70/15/15) with temporal ordering
T = size(Xraw,1);
idxTrainEnd = floor(0.70*T);
idxValEnd   = floor(0.85*T);

Xtrain_raw = Xraw(1:idxTrainEnd, :);
Xval_raw   = Xraw(idxTrainEnd+1:idxValEnd, :);
Xtest_raw  = Xraw(idxValEnd+1:end, :);

%% ---------------------- Missing Value Imputation ------------------------
% Define a threshold (in time steps) for "short gaps" handled by linear interpolation
shortGapMax = 4;  % tweak as needed (5-min steps => up to 20 minutes)
Xtrain_filled = impute_missing(Xtrain_raw, adjList, shortGapMax);
Xval_filled   = impute_missing(Xval_raw,   adjList, shortGapMax);
Xtest_filled  = impute_missing(Xtest_raw,  adjList, shortGapMax);

%% ----------------------------- Normalization ---------------------------
% Fit Min-Max on TRAIN ONLY, apply to val/test (to avoid leakage)
[normParams] = fit_minmax(Xtrain_filled);
Xtrain = apply_minmax(Xtrain_filled, normParams);
Xval   = apply_minmax(Xval_filled,   normParams);
Xtest  = apply_minmax(Xtest_filled,  normParams);

%% ------------------------ Sequence Construction ------------------------
lookback = 12;   % steps (≈ 1 hour if 5-min sampling)
horizon  = 3;    % predict next 3 steps (15 minutes ahead, e.g., 3×5min)
[XTrainSeq, YTrainSeq] = create_sequences(Xtrain, lookback, horizon);
[XValSeq,   YValSeq]   = create_sequences(Xval,   lookback, horizon);
[XTestSeq,  YTestSeq]  = create_sequences(Xtest,  lookback, horizon);

% For Deep Learning Toolbox: sequence data as cell arrays of [features x time]
% Here, features = numSensors, time = lookback (inputs) or horizon (targets).
toSeq = @(Xcell) cellfun(@(A) permute(A,[2 1 3]), Xcell, 'UniformOutput', false);
XTrainDL = toSeq(XTrainSeq);  % each cell: [S x lookback]
YTrainDL = toSeq(YTrainSeq);  % each cell: [S x horizon]
XValDL   = toSeq(XValSeq);
YValDL   = toSeq(YValSeq);
XTestDL  = toSeq(XTestSeq);
YTestDL  = toSeq(YTestSeq);

numSensors = size(Xtrain,2);

%% -------------------------- Build the Network --------------------------
% Architecture: sequenceInput -> GRU(64) -> GRU(64) -> LSTM(64) -> LSTM(64)
% -> AdditiveAttention -> FC(numSensors*horizon) -> reshape to [S x horizon]
numUnits = 64;

layers = [ ...
    sequenceInputLayer(numSensors, "Name","seq_in")

    gruLayer(numUnits, "OutputMode","sequence", "Name","gru1")
    gruLayer(numUnits, "OutputMode","sequence", "Name","gru2")

    lstmLayer(numUnits, "OutputMode","sequence", "Name","lstm1")
    lstmLayer(numUnits, "OutputMode","sequence", "Name","lstm2")

    % Additive attention over time steps (operates across the sequence dimension)
    AdditiveAttentionLayer("attn")

    fullyConnectedLayer(numSensors*horizon, "Name","fc_out")
    functionLayer(@(X) reshape(X, [numSensors horizon 1]), ...
        Formattable=true, Name="reshape_out")

    % Custom MAE regression loss will be supplied in trainingOptions via a dlnetwork
];
layers = [
    sequenceInputLayer(numSensors, "Name","seq_in")
    gruLayer(64,'OutputMode','sequence')
    lstmLayer(64,'OutputMode','sequence')
    %AdditiveAttentionLayer('attn')
    fullyConnectedLayer(numSensors*horizon, "Name","fc_out")
    %regressionLayer
];

%net = dlnetwork(layerGraph(layers));
% Convert to dlnetwork (for custom training with MAE loss & early stopping)
net = dlnetwork(layerGraph(layers));

%% ----------------------- Training Setup (Adam + MAE) -------------------
learnRate = 1e-3;
maxEpochs = 50;
miniBatchSize = 64;
patience = 7;                % early stopping
bestVal = inf; bestNet = net;
wait = 0;
%
dsX = arrayDatastore(XTrainDL,'IterationDimension',1);
dsY = arrayDatastore(YTrainDL,'IterationDimension',1);

% Combine into one datastore
dsTrain = combine(dsX, dsY);
% Now create minibatchqueue


mbqTrain = minibatchqueue(XTrainDL, YTrainDL, ...
    "MiniBatchSize", miniBatchSize, ...
    "MiniBatchFcn", @(x,y) preprocessMiniBatch(x,y), ...
    "MiniBatchFormat", ["CBT","CBT"]); % [channels x batch x time]

mbqVal = minibatchqueue(XValDL, YValDL, ...
    "MiniBatchSize", miniBatchSize, ...
    "MiniBatchFcn", @(x,y) preprocessMiniBatch(x,y), ...
    "MiniBatchFormat", ["CBT","CBT"]);

avgGrad = []; avgSqGrad = [];

for epoch = 1:maxEpochs
    reset(mbqTrain);
    trainLoss = 0; nB = 0;

    while hasdata(mbqTrain)
        [Xb, Yb] = next(mbqTrain);  % Xb: [S x B x lookback], Yb: [S x B x horizon]

        % Evaluate loss and gradients
        [loss, gradients] = dlfeval(@modelLoss, net, Xb, Yb);
        trainLoss = trainLoss + double(gather(extractdata(loss)));
        nB = nB + 1;

        % Adam update
        [net, avgGrad, avgSqGrad] = adamupdate(net, gradients, ...
            avgGrad, avgSqGrad, epoch, learnRate, 0.9, 0.999, 1e-8);
    end

    % Validation
    valLoss = evaluateMAE(net, mbqVal);

    fprintf("Epoch %2d | Train MAE: %.4f | Val MAE: %.4f\n", epoch, trainLoss/nB, valLoss);

    % Early stopping
    if valLoss < bestVal - 1e-4
        bestVal = valLoss; bestNet = net; wait = 0;
    else
        wait = wait + 1;
        if wait >= patience
            fprintf("Early stopping at epoch %d. Best Val MAE = %.4f\n", epoch, bestVal);
            break;
        end
    end
end

net = bestNet;

%% ----------------------------- Evaluation ------------------------------
% Predict on test set
[YPredDL] = predictSequences(net, XTestDL, numSensors, horizon);

% Convert back to numeric arrays [Nseq x horizon x S]
Ytrue = cat(3, YTestDL{:}); Ytrue = permute(Ytrue, [3 2 1]); % [Nseq x horizon x S]
Ypred = cat(3, YPredDL{:}); Ypred = permute(Ypred, [3 2 1]);

% Inverse normalization to original scale for metrics
inv = @(Z) Z.*(normParams.max - normParams.min) + normParams.min;
Ytrue_inv = inv(Ytrue);
Ypred_inv = inv(Ypred);

[MAE, RMSE, MAPE] = compute_metrics(Ytrue_inv, Ypred_inv);
fprintf("TEST  MAE: %.3f | RMSE: %.3f | MAPE: %.2f%%\n", MAE, RMSE, 100*MAPE);

%% --------------------------- Helper Functions --------------------------
function [Xb, Yb] = preprocessMiniBatch(Xc, Yc)
% Convert cell batch to dlarray with format [C B T] (channels=sensors)
    Xb = cat(3, Xc{:});    % [S x lookback x B] -> permute to [S x B x lookback]
    Xb = permute(Xb, [1 3 2]);
    Xb = dlarray(single(Xb), "CBT");

    Yb = cat(3, Yc{:});    % [S x horizon x B] -> permute to [S x B x horizon]
    Yb = permute(Yb, [1 3 2]);
    Yb = dlarray(single(Yb), "CBT");
end

function [loss, gradients] = modelLoss(net, Xb, Yb)
% Forward pass: net outputs [numSensors*horizon x B] before reshape layer result
    Yhat = forward(net, Xb);   % Yhat: [S x horizon x B] packed as dlnetwork output from functionLayer
    % Ensure layout identical to Yb ("CBT")
    loss = maeLoss(Yhat, Yb);  % custom MAE (primary loss)
    gradients = dlgradient(loss, net.Learnables);
end

function mae = evaluateMAE(net, mbq)
    reset(mbq); losses = [];
    while hasdata(mbq)
        [Xb, Yb] = next(mbq);
        Yhat = forward(net, Xb);
        losses(end+1) = gather(extractdata(maeLoss(Yhat, Yb))); %#ok<AGROW>
    end
    mae = mean(losses);
end

function L = maeLoss(Yhat, Ytrue)
% Mean Absolute Error over all sensors, time steps, and batch
    L = mean(abs(Yhat - Ytrue), 'all');
end

function YPredDL = predictSequences(net, XSeq, numSensors, horizon)
    YPredDL = cell(size(XSeq));
    for i = 1:numel(XSeq)
        Xb = XSeq{i};               % [S x lookback]
        Xb = dlarray(single(Xb), "CT");       % C=channels, T=time, batch=1
        Xb = reshape(Xb, [numSensors 1 size(Xb,2)]); % [S x 1 x lookback] -> "CBT"
        Xb = dlarray(Xb, "CBT");
        Yhat = forward(net, Xb);    % [S x horizon x 1]
        YPredDL{i} = stripdims(Yhat);
    end
end
