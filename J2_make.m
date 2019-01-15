%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% This will return Nth column matrix for J2  %%%%%%%%%%%%%%
%%%%%%%%%%%%inputs are ith and jth atom, tersoff parameters, r's, size of the cluster,
%%%%%%%%%%%%type of atoms
function [dVdlambda1, dVdlambda2, dVdbeta, dVdeta, dVdc, dVdd, dVdh] = J2_make(net,param, r, clust_size, type)

%dVdA=0.0;
%dVdB=0.0;
dVdlambda1=0.0;
dVdlambda2=0.0;
dVdbeta=0.0;
dVdeta=0.0;
dVdc=0.0;
dVdd=0.0;
dVdh=0.0;

for i=1:1:clust_size
    for j=1:1:clust_size
        if i==j
            continue;
        end
        %%%%%MILIND
                bi=net.b{1};  %sim(net,r(i,j));
                w=net.IW{1};
                A=w(1)*r(i,j)+bi(1);
                B=w(2)*r(i,j)+bi(2);%param(type(i),type(j),type(j),2);
%         AB2 = sim(net,r(i,j));
%         A = AB2(1);
%         B = AB2(2);
        %%%%%MALSHE
        
        lambda1=param(type(i),type(j),type(j),3);
        lambda2=param(type(i),type(j),type(j),4);
%         lambda3=param(type(i),type(j),type(j),5);
%         alpha=param(type(i),type(j),type(j),6);
        beta=param(type(i),type(j),type(j),7);
        eta=param(type(i),type(j),type(j),8);
%         c=param(type(i),type(j),type(j),9);
%         d=param(type(i),type(j),type(j),10);
%         h=param(type(i),type(j),type(j),11);
        R=param(type(i),type(j),type(j),12);
        D=param(type(i),type(j),type(j),13);        
        [fC]=fc(i,j,r,R,D);
        [fR]=fr(i,j,r,A,lambda1);
        [fA]=fa(i,j,r,B,lambda2);
        %dv/dA
        %dVdA=dVdA+fR;
        %dV/dlambda1
        if fC==0
           dVdlambda1=dVdlambda1-0.0;
        else
            dVdlambda1=dVdlambda1-r(i,j)*fC*fR;
        end
        
        zetaij=0.0;
        dgdc=0.0;
        dgdd=0.0;
        dgdh=0.0;
        for k=1:1:clust_size
            if k==j || k==i
                continue;
            end
%             A=param(type(i),type(j),type(k),1);
%             B=param(type(i),type(j),type(k),2);
%             lambda1=param(type(i),type(j),type(k),3);
%             lambda2=param(type(i),type(j),type(k),4);
%             lambda3=param(type(i),type(j),type(k),5);
%             alpha=param(type(i),type(j),type(k),6);
%             beta=param(type(i),type(j),type(k),7);
%             eta=param(type(i),type(j),type(k),8);
            c=param(type(i),type(j),type(k),9);
            d=param(type(i),type(j),type(k),10);
            h=param(type(i),type(j),type(k),11);
            R=param(type(i),type(j),type(k),12);
            D=param(type(i),type(j),type(k),13);
            [dgdc]   = dgdc+dg_dc(i,j,k,r,c,d,h,R,D);
            [dgdd]   = dgdd+dg_dd(i,j,k,r,c,d,h,R,D);
            [dgdh]   = dgdh+dg_dh(i,j,k,r,c,d,h,R,D);
            [zetaij] = zetaij+zeta(i, j, k, r, c, d, h, R, D);		
        end
        
        beta_eta=beta^eta;
        
        if (zetaij == 0)
            zeta_eta = 0;
        else
            zeta_eta=zetaij^eta;
        end
            
        [bij]=b(zetaij,beta_eta, zeta_eta, eta);
        
        %dV/dB
        %dVdB=dVdB+bij*fB;
        
        %dV/dlambda2
%         if fC==0.0
%             dVdlambda2=dVdlambda2-0.0;
%         else
            dVdlambda2=dVdlambda2-fC*bij*r(i,j)*fA;
%         end
        
        %dV/dbeta
        if fC==0.0
            dVdbeta=dVdbeta-0.0;
        else
            dVdbeta=dVdbeta-0.5*fA*fC*bij*zeta_eta*(beta^(eta-1))/(1+zeta_eta*beta_eta);
        end
        
        %dV/deta
          
        if fC==0.0 || zetaij==0.0
            dVdeta=dVdeta+0.0;
        else
            add_dVdeta=fC*fA*bij*0.5*(log(1+beta_eta*zeta_eta)/eta^2-beta_eta*zeta_eta*(log(beta)+log(zetaij))/(eta*(1+beta_eta*zeta_eta)));
            dVdeta=dVdeta+add_dVdeta;
        end
        
        %common factor
        if zetaij==0
            cf=0;
        else
            if fC==0
                cf=0;
            else
                cf=0.5*fA*fC*bij*beta_eta*(zetaij^(eta-1))/(1+zeta_eta*beta_eta);
            end
        end
        
        
        %dV/dc
        dVdc=dVdc-cf*dgdc;
        
        %dV/dd
        dVdd=dVdd-cf*dgdd;
        
        %dV/dh
        dVdh=dVdh-cf*dgdh;

        
    end
  
end
dVdlambda1=-dVdlambda1/2.0;
dVdlambda2=-dVdlambda2/2.0;
dVdbeta=-dVdbeta/2.0;
dVdeta=-dVdeta/2.0;
dVdc=-dVdc/2.0;
dVdd=-dVdd/2.0;
dVdh=-dVdh/2.0;









