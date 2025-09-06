function params = fit_minmax(Xtrain)
% Fit Min-Max on TRAIN ONLY (Sec. 3.2), then apply to val/test
    params.min = min(Xtrain,[],1,'omitnan');
    params.max = max(Xtrain,[],1,'omitnan');

    % Guard: avoid zero range
    zeroRange = (params.max - params.min) == 0;
    params.max(zeroRange) = params.min(zeroRange) + 1e-6;
end
