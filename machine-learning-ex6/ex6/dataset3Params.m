function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

candidates_number = 8;
multiplier = 3;

sigma_candidate_start_value = 0.01;
sigma_candidates = sigma_candidate_start_value .* ...
    (multiplier .^ [0:candidates_number-1]);

C_candidate_start_value = 0.01;
C_candidates = C_candidate_start_value .* ...
    (multiplier .^ [0:candidates_number-1]);


validation_errors = zeros(candidates_number, candidates_number);

for i = 1:candidates_number
    for j = 1:candidates_number

        s = sigma_candidates(i);
        c = C_candidates(j);

        model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s));
        predictions = svmPredict(model, Xval);

        validation_errors(i, j) = mean(double(predictions ~= yval));
    end
end


[minimums, sigma_indicies] = min(validation_errors);
[minimum, c_index] = min(minimums);

C = C_candidates(c_index);
sigma = sigma_candidates(sigma_indicies(c_index));

% =========================================================================

end
