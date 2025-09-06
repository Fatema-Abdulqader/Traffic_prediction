function [XSeq, YSeq] = create_sequences(X, lookback, horizon)
% Make overlapping sequences:
%   For each start t, input = X[t-lookback+1 : t], output = X[t+1 : t+horizon]
% Returns cell arrays where each cell is [lookback x S] (inputs) or [horizon x S] (targets).

    [T,S] = size(X);
    starts = (lookback):(T - horizon);
    N = numel(starts);

    XSeq = cell(N,1);
    YSeq = cell(N,1);
    for i = 1:N
        t = starts(i);
        Xin = X(t-lookback+1:t, :);   % [lookback x S]
        Yout = X(t+1:t+horizon, :);   % [horizon x S]
        XSeq{i} = single(Xin);        % will be permuted later for DL
        YSeq{i} = single(Yout);
    end
end
