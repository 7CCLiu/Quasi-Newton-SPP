function output = broyd(A, b, c, lambda, gamma, beta, z0, L, T, M, corr)
    [m, d] = size(A);
    z = z0;
    output.norm_grad = zeros(T, 1);
    invG=1/L*eye(d+1);
    Ax=A*z(1:d);
    gz = full_grad(A, b, c, z, m, d , lambda, gamma, beta,Ax,0);
    output.times=zeros(T,1);
    tic;
    for t=1: T
        dz = - invG * hes_vec(A, b, c, z, gz, d, lambda, gamma, beta, m,Ax,0);
        if corr
            r=norm(dz);
            invG = invG / (1 + M* r);
        end

        z = z + dz;
        u = randn(d+1, 1);
        Ax=A*z(1:d);
        Au = hes_vec(A, b, c, z, u, d, lambda, gamma, beta, m,Ax,0);
        Hu = hes_vec(A, b, c, z, Au, d, lambda, gamma, beta, m,Ax, 0);
    
        v = u' * Hu;
        tmp = (invG * Hu) * (u' / v);
        invG = invG - tmp'- tmp + u *((Hu' * tmp+u') /v);
        gz = full_grad(A, b, c, z, m, d , lambda, gamma, beta,Ax,0);

        output.norm_grad(t) = norm(gz);
        timet=toc;
        output.times(t)=norm(timet);
        if mod(t,50)==0
            fprintf('epohces:%d,  grad_norm=%e, time=%f\n', t, output.norm_grad(t), output.times(t));
        end
    end
end

