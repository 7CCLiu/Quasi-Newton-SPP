function output = extra(A, b, c, z0, lambda, gamma, beta, tau, T)
    [m, d] = size(A);
    z=z0;
    output.norm_grad = zeros(T,1);
    output.times=zeros(T,1);
    tic;
    for t=1: T
%         gw = lambda * w + 2 * (1 - p) * (Ap' * (Ap * w - u - (1+y))) / n + 2 * p * (An' * (An * w - v + 1 + y)) / n;
%         gu = lambda * u - 2 * (1 - p) * sum(Ap * w - u) / n;
%         gv = lambda * v - 2 * p * sum(An * w - v) / n;
%         gy = 2 * p * (1 - p) * y + (2 * (1 - p) * sum(Ap * w) - 2 * p * sum(An * w)) / n;
        Ax=A*z(1:d);
        gz = full_grad(A, b, c, z, m, d, lambda, gamma, beta,Ax, 0);
        x=z(1:d);
        y=z(d+1);
        gx=gz(1:d);
        gy=gz(d+1);
        x1=x-tau*gx;
        y1=y+tau*gy;
        z1=[x1;y1];
        Ax1=A*z1(1:d);
        gz1 = full_grad(A, b, c, z1, m, d, lambda, gamma, beta,Ax1, 0);
        gx1=gz1(1:d);
        gy1=gz1(d+1);
        x=x-tau*gx1;
        y=y+tau*gy1;
        z=[x;y];
        timet=toc;
        output.times(t)=norm(timet);
        output.norm_grad(t) = norm(gz);
        if mod(t,50)==0
            fprintf('epohces:%d,  grad_norm=%e, time=%f\n', t, output.norm_grad(t), output.times(t));
        end

    end
    output.weg=z;
end


