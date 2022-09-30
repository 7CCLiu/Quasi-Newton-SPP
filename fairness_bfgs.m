function output = bfgs(A, b, c, lambda, gamma, beta, z0, L, T, M, corr)
    
    [m, d] = size(A);
    z=z0;
    output.norm_grad = zeros(T, 1);;
    output.times=zeros(T,1);
    I=eye(d+1);
    sqr_invG = 1/(L^(1/2))*I;
    Ax=A*z(1:d);
    gz = full_grad(A, b, c, z, m, d, lambda, gamma, beta,Ax,0);
    tic;
    for t=1: T
        dz = - sqr_invG'*(sqr_invG * hes_vec(A, b, c, z, gz, d, lambda, gamma, beta, m,Ax,0));
        if corr
            r = norm(dz);
            sqr_invG = sqr_invG ./ (1 + M * r)^(1/2);
        end

        z = z + dz;
        Ax=A*z(1:d);
        tu = randn(d+1, 1);
        u = sqr_invG' * tu;

        Au = hes_vec(A, b, c, z, u, d, lambda, gamma, beta, m, Ax, 0);
        Hu = hes_vec(A, b, c, z, Au, d, lambda, gamma, beta, m, Ax, 0);

        v = u / ((u' * Hu)^(1/2));
        Bv = hes_vec(A, b, c, z, v, d, lambda, gamma, beta, m,Ax, 0);
        Hv=hes_vec(A, b, c, z, Bv, d, lambda, gamma, beta, m,Ax,0);

        [temp, tmp_r] = qrupdate(I, sqr_invG, -sqr_invG * Hv, v);
        [temp2, sqr_invG] = qrinsert(I, tmp_r, 1, v', 'row');
        sqr_invG = sqr_invG(1:(d+1), :);

        gz = full_grad(A, b, c, z, m, d , lambda, gamma, beta,Ax,0);

        output.norm_grad(t) = norm(gz);
        timet=toc;
        output.times(t)=timet;
        if mod(t,50)==0
            fprintf('epohces:%d,  grad_norm=%e, time=%f\n', t, output.norm_grad(t), output.times(t));
        end

    end
end
