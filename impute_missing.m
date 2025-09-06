function Xfilled = impute_missing(X, adjList, shortGapMax)
% Missing-value handling:
% 1) Linear interpolation for short gaps (<= shortGapMax).
% 2) Spatial neighbor averaging for longer gaps: x_i(t) = mean_j in N(i) x_j(t)  (Eq. 2)

    Xfilled = X;

    % --- 1) Linear interpolation per sensor for short gaps ---
    for s = 1:size(X,2)
        x = Xfilled(:,s);
        miss = isnan(x);
        if any(miss)
            % Find all contiguous missing segments
            d = diff([0; miss; 0]);
            segStart = find(d==1);
            segEnd   = find(d==-1)-1;

            for k = 1:numel(segStart)
                L = segEnd(k) - segStart(k) + 1;
                if L <= shortGapMax
                    i1 = segStart(k)-1;  % index before gap
                    i2 = segEnd(k)+1;    % index after gap
                    if i1>=1 && i2<=numel(x) && ~isnan(x(i1)) && ~isnan(x(i2))
                        t = (1:L)';
                        x(segStart(k):segEnd(k)) = x(i1) + (t/(L+1))*(x(i2)-x(i1));
                    end
                end
            end
            Xfilled(:,s) = x;
        end
    end

    % --- 2) Spatial neighbor averaging for remaining (long) gaps ---
    [T,S] = size(Xfilled);
    for t = 1:T
        missS = find(isnan(Xfilled(t,:)));
        if isempty(missS), continue; end
        for s = missS
            neighbors = adjList{s};
            neighbors = neighbors(neighbors>=1 & neighbors<=S);
            if ~isempty(neighbors)
                vals = Xfilled(t, neighbors);
                vals = vals(~isnan(vals));
                if ~isempty(vals)
                    Xfilled(t,s) = mean(vals);
                end
            end
        end
    end

    % Optional: remaining NaNs (edge cases) -> forward/backward fill
    for s = 1:size(Xfilled,2)
        x = Xfilled(:,s);
        if any(isnan(x))
            x = fillmissing(x,'previous');
            x = fillmissing(x,'next');
            Xfilled(:,s) = x;
        end
    end
end
