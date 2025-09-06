function [MAE, RMSE, MAPE] = compute_metrics(Ytrue, Ypred)
% Inputs: [Nseq x horizon x S]
    err = Ypred - Ytrue;
    MAE  = mean(abs(err), 'all');
    RMSE = sqrt(mean(err.^2, 'all'));

    denom = max(abs(Ytrue), 1e-6);       % avoid div by zero
    MAPE = mean(abs(err)./denom, 'all');  % fraction; multiply by 100 for %
end
