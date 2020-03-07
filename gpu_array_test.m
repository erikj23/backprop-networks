tic
A = rand(2^16, 1);
B = fft(A);
toc

tic
A = gpuArray(rand(2^16, 1));
B = fft(A);
toc