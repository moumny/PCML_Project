function weights= initializeWeights(M, H1, H2)
% furnishes a set of initialized vector
% made following the instructions of page 9 of description 
% output : 
% initialized weights (column vector form)

W1L= randn(H1,M)/M;
B1L= randn(H1,1)/M;
W1R= randn(H1, M)/M;
B1R= randn(H1,1)/M;

W2L= randn(H2, H1)/H1;
B2L= randn(H2,1)/H1;
W2LR= randn(H2, 2*H1)/2/H1;
B2LR= randn(H2, 1)/2/H1;
W2R= randn(H2, H1)/H1;
B2R= randn(H2,1)/H1;

W3= randn(1,H2)/H2;
B3= randn(1,1)/H2;

weights=weightsToVector( M, H1, H2, W1L, B1L, W1R, B1R, W2L, B2L, W2LR, B2LR, W2R, B2R, W3, B3 );
end

