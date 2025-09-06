function Xn = apply_minmax(X, params)
    Xn = (X - params.min) ./ (params.max - params.min);
end
