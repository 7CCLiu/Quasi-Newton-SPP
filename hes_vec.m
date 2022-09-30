function [hv]= hes_vec(A, b, c, z, v, d, lambda, gamma, beta, m,Ax,gpu)
    x=z(1:d);
    y=z(d+1);
    vx=v(1:d);
    vy=v(d+1);
    p1=1./(1+exp(b.*Ax));
    p2=1./(1+exp(c.*(Ax)*y));
    Avx=A*vx;
    hxx=A'*(p1.*(1-p1)/m.*(Avx)-p2.*(1-p2)*beta/m*y^2.*(Avx))+2*lambda*vx;
    hxy=-A'*((Ax)*y.*p2.*(1-p2)*beta/m)*vy+A'*(p2*beta/m.*c)*vy;
    hyx=sum(-beta/m*p2.*(1-p2).*(Ax)*y.*(Avx))+sum(p2*beta/m.*c.*Avx);
    hyy=(-sum(p2.*(1-p2)/m*beta.*(Ax).^2)-2*gamma)*vy;

    hv=[hxx+hxy;hyy+hyx];
    if gpu==1 
        hv=gpuArray(hv);
    end

end
