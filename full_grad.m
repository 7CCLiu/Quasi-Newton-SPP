function gz = full_grad(A, b, c, z, m, d, lambda, gamma, beta,Ax,gpu)
    x= z(1:d);
    y= z(d+1);
    p1=0.5*(1+tanh(-0.5*(b.*Ax)));
    p2=0.5*(1+tanh(-0.5*(y*c.*Ax)));
    gx=-A'*(p1.*b/m)+A'*(y*beta/m*c.*p2)+2*lambda*x;
    gy=sum((beta/m)*p2.*c.*(Ax))-2*gamma*y;
    gz=[gx;gy];
    if gpu
        gz=gpuArray(gz);
    end
end



