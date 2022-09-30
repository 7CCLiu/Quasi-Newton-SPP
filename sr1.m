function output = sr1(A, b, c, lambda, gamma, beta, z0, L, T, M, corr)
    [m, d] = size(A);
    z = z0;
    output.norm_grad = zeros(T, 1);
    invG=1/L*eye(d+1);
    output.times=zeros(T,1);
    tic;
    Ax=A*z(1:d);
    gz = full_grad(A, b, c, z, m, d , lambda, gamma, beta,Ax,0);

    for t=1: T
        dz = - invG * hes_vec(A, b, c, z, gz, d, lambda, gamma, beta, m,Ax,0);
        if corr
            r = norm(dz);
            invG = invG / (1 + M * r);
        end

        z = z + dz;
        Ax=A*z(1:d);
        u = randn(d+1, 1);
        Au = hes_vec(A, b, c, z, u, d, lambda, gamma, beta, m,Ax,0);
        Hu = hes_vec(A, b, c, z, Au, d, lambda, gamma, beta, m,Ax,0);

        v = invG * Hu;
        hes_temp= hes_vec(A, b, c, z, u-v, d, lambda, gamma, beta, m,Ax,0);
        temp=u'*hes_vec(A, b, c, z, hes_temp, d, lambda, gamma, beta, m,Ax,0)+1e-30;

        invG = invG + (u - v)*((u - v)' / temp );
        gz=full_grad(A, b, c, z, m, d , lambda, gamma, beta,Ax,0);
        timet=toc;
        output.times(t)=timet;
        output.norm_grad(t) = norm(gz);
        if mod(t,50)==0
            fprintf('epohces:%d,  grad_norm=%e, time=%f\n', t, output.norm_grad(t), output.times(t));
        end
    end
end

