function [data, dataTime, adjList] = load_pems_example()
% Replace this with a real PeMS loader. Here we synthesize a tiny example.
    T = 1000; S = 8;
    t = (1:T)'; dataTime = t; %#ok<NASGU>
    base = sin(2*pi*(t/288)) + 0.1*randn(T,1);  % daily pattern (5-min slots -> 288/day)
    X = base .* (0.8 + 0.4*rand(1,S)) + 0.05*randn(T,S);
    % Insert NaNs as missing
    for s=1:S
        idx = randi([50 T-50],1,3);
        for k=1:numel(idx)
            L = randi([2 8],1);  % short or long gaps
            X(idx(k):idx(k)+L, s) = NaN;
        end
    end
    data.speed = X;
    data.flow  = X*100 + 10*randn(T,S);
    data.occ   = abs(X*10 + randn(T,S));

    % One-hop physical neighbors (line topology as an example)
    adjList = cell(1,S);
    for i=1:S
        neigh = [i-1 i+1];
        neigh = neigh(neigh>=1 & neigh<=S);
        adjList{i} = neigh;  % direct upstream/downstream (Sec. 3.2)
    end
end
