import numpy as np
from scipy.optimize import brentq, newton, root, fsolve
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class OLG:

    def __init__(self, **kwargs): 
        self.baseline()
        self.update_parameters(kwargs)
        self.update_parameters_comp()
    
    #1. PARAMETERS
    def baseline(self):
        
        #Population
        self.m0 = np.array([0.2542, 0.2444, 0.2013, 0.1319, 0.0907, 0.0772]) #En proporciones 2000
        # self.m0 = np.array([2.542, 2.444, 2.013, 1.319, 0.907, 0.772]) #En proporciones 2000
                
        self.m1 = np.array([0.211, 0.208, 0.182, 0.166, 0.126, 0.108]) ##En proporciones 2020
        
        # self.m0 = np.array([0.250, 0.241, 0.198, 0.130, 0.089, 0.076]) #En proporciones 2000
        
#         self.m1 = np.array([0.30, 0.295, 0.258, 0.235, 0.179, 0.153]) ##En proporciones 2020        
        self.m1 = self.m0 + 0.0001
        # self.m1 = np.ones(6)/6
        self.m = self.m0 #first value
        self.N0 = sum(self.m0)
        self.N1 = sum(self.m1)
        self.N = self.N0 #initial value
        
        #A growht
        self.ga = 0.01 #0.018
        self.A = 1.
        
        #Probability of survive
        # self.ϕ0 = np.array([1., 0.99, 0.98, 0.97, 0.96, 0.96])
        # self.ϕ0 = np.array([1., 1., 1., 1., 0.96, 0.96])
        self.ϕ0 = np.ones(6)
        # self.ϕ1 = np.array([1., 1., 1., 1., 0.96, 0.96])
        # self.ϕ1 = np.array([1., 0.995, 0.989, 0.977, 0.948, 0.01])
        self.ϕ1 = np.ones(6)
        self.ϕ = self.ϕ0 #first value
        
        #Efficiency
        self.efage = np.array([0.726, 0.774, 0.722, 0.7])
        # efage = np.ones(4)        
        self.ef = np.array([[1.43], [0.57]]) #1 alta eficiencia, 2 baja eficiencia
        # self.ef = np.array([[1], [1]])
        self.eμ = 2 #Para calibración de productividad.
        
        self.me1 = 0.5
        self.me2 = 0.5
        
        #Utility
        self.γ = 1.32
        self.η = 2.0
        self.β = 0.9        #discount factor
        
        #Production
        self.α = 0.3 #0.35
        self.δ= 0.4 #0.08
        
        #Taxes: goes to goverment consumption and social pension  
        self.τ_w = 0.1#0.15  #0.248 
        self.τ_r = 0.05#0.2# 0.429   
        self.τ_c = 0.1#0.05 #
        
        #Social security contribution
        self.μ = 1. #proportion between capitalization and pay as you go. 1=capitalization
        self.τ_ic0 =  0.00
        self.τ_cc0 =  0.0
        self.τ_pg0 = 0.
        
        self.τ_ic1 =  0.00
        self.τ_cc1 =  0.0
        self.τ_pg1 = 0.        
        self.pen_cc_2 = 0.1
        self.pen_cc_2_0 = 0.0
        self.pen_cc_2_1 = 0.0
        
        #Government
        self.g = 0#0.2#0.1
        self.ν = 0.0
    
        self.ρ = 0.8        #updating parameter for method 1
        #Parameters

        self.r = 0.2        #initial value of the interest rate
        self.s=2            #coefficient of relative risk aversion
#         self.α=0.3          #production elasticity of capital

        self.T = 4 #NUEVOS: edades laborales
        self.TR=2  #NUEVOS: edades retiro        
        
        self.rep0 = 0.2   #replacement rate (Pay as you go and AFP) ss0
        self.rep1 = 0.3   #replacement rate (Pay as you go and AFP) ss1
        self.τ_afp0 = 0.1
        self.τ_afp1 = 0.0
               
        self.dep=0.4        #rate of depreciation

#         self.τ = self.ζ0/((self.T/self.TR)+self.ζ0)   #income tax rate
        self.τ0 = 0   #income tax rate
        self.τ1 = 0.1   #income tax rate    
        self.ζ = (self.T/self.TR)*(self.τ0/(1-self.τ0))         #replacement ratio pay as you go ss0
 
        self.gam=2    #disutility from working
#         self.kmax=1         #upper limit of capital grid
#         self.kinit=0        #
#         self.na=101         #number of grid points on assets
#         self.a=           #asset grid
        self.ψ=0.001        #parameter of utility function
        self.tol=0.001      #percentage deviation of final solution
        self.tolk=0.00001    #percentage deviation of final solution for k_1
#         self.nq1=30         #number of iterations over k^1
        self.nt=20          #number of transition periods
        self.nqt=20         #maximum number of iterations over transition of K_t,N_t
        self.tolt=0.00001    #tolerance with regard to K_t,N_t 
        
        #Initialization
        self.nbar=0.3
        self.kbar=(self.α/(self.r+self.dep))**(1/(1-self.α))*self.nbar
        self.kold=100
        self.nold=2
        self.cy = 0.8
        self.n0=0.3        
        
        #Agent's policy function in steady state
        self.aopt = np.zeros(6)
        self.copt = np.zeros(6)
        self.nopt = 0.3 * np.ones(4)
        self.w = 0
#         self.r = 0
        self.pen = 0
        self.i = 0
        self.wseq = np.zeros(6)
        self.rseq = np.zeros(6)
        self.penseq = np.zeros(6)
        self.τseq = np.zeros(6)

        
    def update_parameters(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def update_parameters_comp(self, ):
        rep0, T, TR, α, r, dep, nbar = self.rep0, self.T, self.TR, self.α, self.r, self.dep, self.nbar
        τ0 = self.τ0
        efage, ef, eμ = self.efage, self.ef, self.eμ
        self.ζ = (T/TR)*(τ0/(1-τ0))
        self.τ = rep0 / ((T/TR) + rep0)
        self.nt = (T + TR)*5
        self.nqt = (T + TR)*5
        self.kbar = (α/(r+dep))**(1/(1-α))*nbar
        self.e = eμ*efage * ef 
        
    #2. ADDITIVE SEQUENCE  
    def seqa(self, x, inc, n):
    #x: start: sepecify the first element
    #inc: specify increment
    #n: specify the number of elements in the sequence
        output = []
        for i in range(0,n):
            if i == 0:
                output.append(x)
            else:
                x = x + inc
                output.append(x)

        return np.array(output)
    
    
    #3. UTILITY
    def u(self, c, l):
        # A, η, γ = self.A, self.η, self.γ
        s , ψ, gam, A = self.s, self.ψ, self.gam, self.A
        # return ((c**γ*A**γ*(1-l)**(1-γ))**(1-η))/(1-η)   
        return (((A*c + ψ)*l**gam)**(1-s)-1)/(1-s)
        
    #4. MARGINAL UTILITY    
#     def uc(self, c, l):
#         A, η, γ = self.A, self.η, self.γ
#         return (γ*(A**γ*(1-l)**(1-γ)*c**γ)**(1-η))/c
    
    def uc(self, x, y):
        s , ψ, gam, A = self.s, self.ψ, self.gam, self.A
#        return ((x+ψ)**(-s) * y**(gam*(1-s)))     
        return A*y**gam*(y**gam * (A*x + ψ))**(-s)

#     def ul(self, c, l):
#         A, η, γ = self.A, self.η, self.γ
#         return ((γ-1)*(A**γ*c**γ*(1-l)**(1-γ))**(1-η))/(1-l)    


    def ul(self, x, y):
        s , ψ, gam, A= self.s, self.ψ, self.gam, self.A
#        return (gam*(x+ψ)**(1-s) * y**(gam*(1-s)-1))
        return (gam*(y**gam*(A*x + ψ))**(1-s))/y
    
    #Residual function
    def rftr_SS(self, x, τ_ic, τ_cc, τ_pg):
        β, γ, η, T, TR, δ, α, ζ, e, ga, τ_w, g, N, m, ϕ, τ_r, A, τ_c = self.β, self.γ, self.η, self.T, self.TR, self.δ, self.α, self.ζ, self.e, self.ga, self.τ_w, self.g, self.N, self.m, self.ϕ, self.τ_r, self.A, self.τ_c
        gam, ψ = self.gam, self.ψ
        μ, ν = self.μ, self.ν
        me1, me2 = self.me1, self.me2
        
        y = np.zeros(21)  #equations to solve    

        ω2_1, ω3_1, ω4_1, ω5_1, ω6_1, l1_1, l2_1, l3_1, l4_1 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8] 
        ω2_2, ω3_2, ω4_2, ω5_2, ω6_2, l1_2, l2_2, l3_2, l4_2 = x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17] 
        L, K, C = x[18], x[19], x[20]
        
        #Prices
        w = (1-α)*K**α  
        r = (α)*K**(α-1) - δ        
        
        rep = 2*((1-μ)*τ_pg)/(1-(1-μ)*τ_pg)  # Actualizar el 2, parece que es T/TR
        pen = rep*(1-(1-μ)*τ_pg)*w*L*((T+TR)/T) #
        
        #Government
        Ω1 = ω2_1*m[1]*ϕ[1]*me1 + ω3_1*m[2]*ϕ[2]*me1 + ω4_1*m[3]*ϕ[3]*me1 + ω5_1*m[4]*ϕ[4]*me1 + ω6_1*m[5]*ϕ[5]*me1
        Ω2 = ω2_2*m[1]*ϕ[1]*me2 + ω3_2*m[2]*ϕ[2]*me2 + ω4_2*m[3]*ϕ[3]*me2 + ω5_2*m[4]*ϕ[4]*me2 + ω6_2*m[5]*ϕ[5]*me2
                
        Tax = τ_w*w*L/(A*L) + τ_r*r*Ω1/L + τ_r*r*Ω2/L + τ_c*C #C ya viene estacionario (por eso no es /L)  
        Beq = (ω2_1*m[1]*(1-ϕ[1])*me1 + ω3_1*m[2]*(1-ϕ[2])*me1 + ω4_1*m[3]*(1-ϕ[3])*me1 + ω5_1*m[4]*(1-ϕ[4])*me1 + ω6_1*m[5]*(1-ϕ[5])*me1)/L + (ω2_2*m[1]*(1-ϕ[1])*me2 + ω3_2*m[2]*(1-ϕ[2])*me2 + ω4_2*m[3]*(1-ϕ[3])*me2 + ω5_2*m[4]*(1-ϕ[4])*me2 + ω6_2*m[5]*(1-ϕ[5])*me2)/L 
    
        tr = ν*(Tax + Beq - g*K**α)*L/(m[4]*me2 + m[5]*me2) #Agregar la proporción me sube la transferencia
        
        #Pensions
        """Individual capitalization (4x2)"""
        #High efficiency
        π_ic_1_1 = μ*τ_ic*w*l1_1*e[0][0]
        π_ic_2_1 = μ*τ_ic*w*l2_1*e[0][1] + (1+r)*π_ic_1_1
        π_ic_3_1 = μ*τ_ic*w*l3_1*e[0][2] + (1+r)*π_ic_2_1
        π_ic_4_1 = μ*τ_ic*w*l4_1*e[0][3] + (1+r)*π_ic_3_1   
        
        #Low efficiency
        π_ic_1_2 = μ*τ_ic*w*l1_2*e[1][0]
        π_ic_2_2 = μ*τ_ic*w*l2_2*e[1][1] + (1+r)*π_ic_1_2
        π_ic_3_2 = μ*τ_ic*w*l3_2*e[1][2] + (1+r)*π_ic_2_2
        π_ic_4_2 = μ*τ_ic*w*l4_2*e[1][3] + (1+r)*π_ic_3_2     
        
        """ ACA: ACTUALIZAR denominador pen_cc_1 con dos productividades """
        # """Collective capitalization (4x1)"""         
        π_cc_1_1 = μ*τ_cc*w*l1_1*e[0][0] 
        π_cc_2_1 = μ*τ_cc*w*l2_1*e[0][1] + (1+r)*π_cc_1_1 
        π_cc_3_1 = μ*τ_cc*w*l3_1*e[0][2] + (1+r)*π_cc_2_1 
        π_cc_4_1 = μ*τ_cc*w*l4_1*e[0][3] + (1+r)*π_cc_3_1     
        
        π_cc_1_2 = μ*τ_cc*w*l1_2*e[1][0]
        π_cc_2_2 = μ*τ_cc*w*l2_2*e[1][1] + (1+r)*π_cc_1_2 
        π_cc_3_2 = μ*τ_cc*w*l3_2*e[1][2] + (1+r)*π_cc_2_2 
        π_cc_4_2 = μ*τ_cc*w*l4_2*e[1][3] + (1+r)*π_cc_3_2     
          
        """ Agencia de pensiones """
        #Individual capitalization 
        ing_ic1 = π_ic_4_1*me1*m[4]
        ing_ic2 = π_ic_4_2*me2*m[4]        
        
        pen_ic_1 = ing_ic1/(me1*(m[4]+m[5]))
        pen_ic_2 = ing_ic2/(me2*(m[4]+m[5]))        
        
        # Collective capitalization 
        pen_cc_2 = self.pen_cc_2 #ahora es valor fijo #(π_cc_4_1 + π_cc_4_2)/(2*TR)
        ing_cc1 = π_cc_4_1*me1*m[4]
        ing_cc2 = π_cc_4_2*me2*m[4]
        
        cost_cc2_4 = pen_cc_2*me2*m[4]
        cost_cc2_5 = pen_cc_2*me2*m[5]
        
        FR_cc = (ing_cc1+ing_cc2) - (cost_cc2_4+cost_cc2_5)
        pen_cc_1 = FR_cc/(me1*(m[4]+m[5]))  #(π_cc_4_1 + π_cc_4_2)/(2*TR)  
              
            
        """Social pensions"""
        pen_sp_1 = 0.
        pen_sp_2 = tr/TR            
        
        #Consumption
        c1_1 = ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l1_1*e[0][0] - ω2_1*(1+ga))/(1+τ_c)
        c2_1 = ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l2_1*e[0][1] + (1+(1-τ_r)*r)*ω2_1 - ω3_1*(1+ga))/(1+τ_c)
        c3_1 = ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l3_1*e[0][2] + (1+(1-τ_r)*r)*ω3_1 - ω4_1*(1+ga))/(1+τ_c)
        c4_1 = ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l4_1*e[0][3] + (1+(1-τ_r)*r)*ω4_1 - ω5_1*(1+ga))/(1+τ_c)
        c5_1 = (pen_ic_1 + pen_cc_1 + pen_sp_1 + pen + (1+(1-τ_r)*r)*ω5_1 - ω6_1*(1+ga))/(1+τ_c)
        c6_1 = (pen_ic_1 + pen_cc_1 + pen_sp_1 + pen + (1+(1-τ_r)*r)*ω6_1)/(1+τ_c)     
        
        c1_2 = ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l1_2*e[1][0] - ω2_2*(1+ga))/(1+τ_c)
        c2_2 = ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l2_2*e[1][1] + (1+(1-τ_r)*r)*ω2_2 - ω3_2*(1+ga))/(1+τ_c)
        c3_2 = ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l3_2*e[1][2] + (1+(1-τ_r)*r)*ω3_2 - ω4_2*(1+ga))/(1+τ_c)
        c4_2 = ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l4_2*e[1][3] + (1+(1-τ_r)*r)*ω4_2 - ω5_2*(1+ga))/(1+τ_c)
        c5_2 = (pen_ic_2 + pen_cc_2 + pen_sp_2 + pen + (1+(1-τ_r)*r)*ω5_2 - ω6_2*(1+ga))/(1+τ_c)
        c6_2 = (pen_ic_2 + pen_cc_2 + pen_sp_2 + pen + (1+(1-τ_r)*r)*ω6_2)/(1+τ_c)     
        
        #Labor supply
        y[0] = self.ul(c1_1, 1-l1_1)/self.uc(c1_1, 1-l1_1) - ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*e[0][0])/(1+τ_c)
        y[1] = self.ul(c2_1, 1-l2_1)/self.uc(c2_1, 1-l2_1) - ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*e[0][1])/(1+τ_c)
        y[2] = self.ul(c3_1, 1-l3_1)/self.uc(c3_1, 1-l3_1) - ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*e[0][2])/(1+τ_c)
        y[3] = self.ul(c4_1, 1-l4_1)/self.uc(c4_1, 1-l4_1) - ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*e[0][3])/(1+τ_c)

        y[4] = self.ul(c1_2, 1-l1_2)/self.uc(c1_2, 1-l1_2) - ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*e[1][0])/(1+τ_c)
        y[5] = self.ul(c2_2, 1-l2_2)/self.uc(c2_2, 1-l2_2) - ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*e[1][1])/(1+τ_c)
        y[6] = self.ul(c3_2, 1-l3_2)/self.uc(c3_2, 1-l3_2) - ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*e[1][2])/(1+τ_c)
        y[7] = self.ul(c4_2, 1-l4_2)/self.uc(c4_2, 1-l4_2) - ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*e[1][3])/(1+τ_c)

        #####################################################3
        #Euler equation
        y[8]  = (1+ga)*ϕ[0]/((β*(1+(1-τ_r)*r))*ϕ[1]) - self.uc(c2_1,1-l2_1)/self.uc(c1_1,1-l1_1) 
        y[9]  = (1+ga)*ϕ[0]/((β*(1+(1-τ_r)*r))*ϕ[2]) - self.uc(c3_1,1-l3_1)/self.uc(c2_1,1-l2_1) 
        y[10] = (1+ga)*ϕ[0]/((β*(1+(1-τ_r)*r))*ϕ[3]) - self.uc(c4_1,1-l4_1)/self.uc(c3_1,1-l3_1) 
        y[11] = (1+ga)*ϕ[0]/((β*(1+(1-τ_r)*r))*ϕ[4]) - self.uc(c5_1,1)  /self.uc(c4_1,1-l4_1) 
        y[12] = (1+ga)*ϕ[0]/((β*(1+(1-τ_r)*r))*ϕ[5]) - self.uc(c6_1,1)  /self.uc(c5_1,1) 


        y[13] = (1+ga)*ϕ[0]/((β*(1+(1-τ_r)*r))*ϕ[1]) - self.uc(c2_2,1-l2_2)/self.uc(c1_2,1-l1_2) 
        y[14] = (1+ga)*ϕ[0]/((β*(1+(1-τ_r)*r))*ϕ[2]) - self.uc(c3_2,1-l3_2)/self.uc(c2_2,1-l2_2) 
        y[15] = (1+ga)*ϕ[0]/((β*(1+(1-τ_r)*r))*ϕ[3]) - self.uc(c4_2,1-l4_2)/self.uc(c3_2,1-l3_2) 
        y[16] = (1+ga)*ϕ[0]/((β*(1+(1-τ_r)*r))*ϕ[4]) - self.uc(c5_2,1)  /self.uc(c4_2,1-l4_2) 
        y[17] = (1+ga)*ϕ[0]/((β*(1+(1-τ_r)*r))*ϕ[5]) - self.uc(c6_2,1)  /self.uc(c5_2,1) 
                
        #Aggregate Capital and labor
        """Aggregate labor"""
        y[18] = (e[0][0]*l1_1*m[0]*me1 + e[0][1]*l2_1*m[1]*me1 + e[0][2]*l3_1*m[2]*me1 + e[0][3]*l4_1*m[3]*me1) + (e[1][0]*l1_2*m[0]*me2 + e[1][1]*l2_2*m[1]*me2 + e[1][2]*l3_2*m[2]*me2 + e[1][3]*l4_2*m[3]*me2) - L 
        
        """Aggregate wealth"""
        Ω1 = ω2_1*m[1]*ϕ[1]*me1 + ω3_1*m[2]*ϕ[2]*me1 + ω4_1*m[3]*ϕ[3]*me1 + ω5_1*m[4]*ϕ[4]*me1 + ω6_1*m[5]*ϕ[5]*me1
        Ω2 = ω2_2*m[1]*ϕ[1]*me2 + ω3_2*m[2]*ϕ[2]*me2 + ω4_2*m[3]*ϕ[3]*me2 + ω5_2*m[4]*ϕ[4]*me2 + ω6_2*m[5]*ϕ[5]*me2

        """Aggregate pension funds"""
        #Individual capitalization
        cost_ic1_4 = pen_ic_1*me1*m[4]
        cost_ic1_5 = pen_ic_1*me1*m[5]
        
        cost_ic2_4 = pen_ic_2*me2*m[4]
        cost_ic2_5 = pen_ic_2*me2*m[5]
        
        Π_ic_1_5 = ing_ic1 - cost_ic1_4
        Π_ic_1_6 = Π_ic_1_5 - cost_ic1_5 
        
        Π_ic_2_5 = ing_ic2 - cost_ic2_4
        Π_ic_2_6 = Π_ic_2_5 - cost_ic2_5 
        
        Π_ic_56 = Π_ic_1_5 + Π_ic_2_5 +  Π_ic_1_6 +  Π_ic_2_6 #Π_cc_1_6 +  Π_cc_2_6  =0      
        
        Π_ic = (π_ic_1_1*m[0]*me1 + π_ic_2_1*m[1]*me1 + π_ic_3_1*m[2]*me1 + π_ic_4_1*m[3]*me1) + (π_ic_1_2*m[0]*me2 + π_ic_2_2*m[1]*me2 + π_ic_3_2*m[2]*me2 + π_ic_4_2*m[3]*me2) + (Π_ic_56)
        # Π_ic = (π_ic_1_1*m[0]*me1 + π_ic_2_1*m[1]*me1 + π_ic_3_1*m[2]*me1 + π_ic_4_1*m[3]*me1) + (π_ic_1_2*m[0]*me2 + π_ic_2_2*m[1]*me2 + π_ic_3_2*m[2]*me2 + π_ic_4_2*m[3]*me2) 

        
        #Collective capitalization
        cost_cc1_4 = pen_cc_1*me1*m[4]
        cost_cc1_5 = pen_cc_1*me1*m[5]        
        Π_cc_1_5 = ing_cc1 - cost_cc1_4
        Π_cc_1_6 = Π_cc_1_5 - cost_cc1_5 
        
        Π_cc_2_5 = ing_cc2 - cost_cc2_4
        Π_cc_2_6 = Π_cc_2_5 - cost_cc2_5 
        
        Π_cc_56 = Π_cc_1_5 + Π_cc_2_5 +  Π_cc_1_6 +  Π_cc_2_6 #Π_cc_1_6 +  Π_cc_2_6  =0      
          
        Π_cc = (π_cc_1_1*m[0]*me1 + π_cc_2_1*m[1]*me1 + π_cc_3_1*m[2]*me1 + π_cc_4_1*m[3]*me1) + (π_cc_1_2*m[0]*me2 + π_cc_2_2*m[1]*me2 + π_cc_3_2*m[2]*me2 + π_cc_4_2*m[3]*me2) + (Π_cc_56) 
                
        """Aggregate capital"""
        # y[19] = 1/L*(Ω1 + Ω2) - K  #SIN PENSIONES 
        y[19] = 1/L*(Ω1 + Ω2 + Π_ic + Π_cc) - K  
        
        """Aggregate consumption"""
        y[20] = 1/L * ((c1_1*m[0]*me1 +c2_1*m[1]*me1 +c3_1*m[2]*me1 +c4_1*m[3]*me1 +c5_1*m[4]*me1 +c6_1*m[5]*me1 ) + (c1_2*m[0]*me2 +c2_2*m[1]*me2 +c3_2*m[2]*me2 +c4_2*m[3]*me2 +c5_2*m[4]*me2 +c6_2*m[5]*me2))  - C
    
    
        return y     

    #7. GET STEADY STATE
    def getss(self, τ_ic, τ_cc, τ_pg, prod=False):
        α, δ, ζ, kbar, nbar, cy, n0, T, TR, τ_w, cy, m, φ = self.α, self.δ, self.ζ, self.kbar, self.nbar, self.cy, self.n0, self.T, self.TR, self.τ_w, self.cy, self.m, self.φ
        μ, A, τ_r, τ_w, τ_c, g, ν, ga = self.μ, self.A, self.τ_r, self.τ_w, self.τ_c, self.g, self.ν, self.ga
        me1, me2 = self.me1, self.me2
        #Initial values
        #Prices
        w = (1-α)*kbar**α
        r = α*kbar**(α-1)  - δ
        
        ω2=(1-cy)*(1-τ_w-μ*τ_ic-μ*τ_cc)*w*n0
        ω3=(1-cy)*((1-τ_w-μ*τ_ic-μ*τ_cc)*w*n0 + r*ω2) + ω2
        ω4=(1-cy)*((1-τ_w-μ*τ_ic-μ*τ_cc)*w*n0 + r*ω3) + ω3
        ω5=(1-cy)*((1-τ_w-μ*τ_ic-μ*τ_cc)*w*n0 + r*ω4) + ω4   
        ω6=ω5/2
        
        x0 = np.array([ω2, ω3, ω4, ω5, ω6, n0, n0, n0, n0, 
                       ω2, ω3, ω4, ω5, ω6, n0, n0, n0, n0,
                       nbar, kbar, 1.5])
        
        
        l_in = np.array([0.40514584, 0.34107888, 0.27011178, 0.19150139] )
        ω_in = np.array([0.0353557 , 0.06376005, 0.07722532, 0.06203296, 0.04971866])
        Kbar_in = 0.27#1546
        Lbar_in = 0.37#5319 
        Cbar_in = 0.26#4736
        x0 = np.concatenate([ω_in, l_in, ω_in, l_in, [Lbar_in], [Kbar_in], [Cbar_in]])
                
        
        
      
        #Solve system of equations
        sol = fsolve(self.rftr_SS, x0, args=(τ_ic, τ_cc, τ_pg), xtol=1.49012e-10)
        # sol = self.rftr_SS(x0, τ_ic, τ_cc)     

        ω1 = sol[0:5]
        l1 = sol[5:9]
        
        ω2 = sol[9:14]
        l2 = sol[14:18]
        
        Lbar = sol[18] #sol[19]                
        Kbar = sol[19] #sol[18]
        Cbar = sol[20]

              
        #################Precios, pensiones y consumo######################
        e = self.e
        w = (1-α)*Kbar**α 
        r = (α)*Kbar**(α-1) - δ
        
        #Pay as you go       
        rep = 2*(1-μ)*τ_pg/(1-(1-μ)*τ_pg)  #
        pen = rep*(1-(1-μ)*τ_pg)*w*Lbar*((T+TR)/T) #
        
        #Capitalization individual
        π_ic_1_1 = μ*τ_ic*w*l1[0]*e[0][0]
        π_ic_2_1 = μ*τ_ic*w*l1[1]*e[0][1] + (1+r)*π_ic_1_1
        π_ic_3_1 = μ*τ_ic*w*l1[2]*e[0][2] + (1+r)*π_ic_2_1
        π_ic_4_1 = μ*τ_ic*w*l1[3]*e[0][3] + (1+r)*π_ic_3_1   
        
        Π_ic_1 = [π_ic_1_1, π_ic_2_1, π_ic_3_1, π_ic_4_1]
        
        π_ic_1_2 = μ*τ_ic*w*l2[0]*e[1][0]
        π_ic_2_2 = μ*τ_ic*w*l2[1]*e[1][1] + (1+r)*π_ic_1_2
        π_ic_3_2 = μ*τ_ic*w*l2[2]*e[1][2] + (1+r)*π_ic_2_2
        π_ic_4_2 = μ*τ_ic*w*l2[3]*e[1][3] + (1+r)*π_ic_3_2   
        
        Π_ic_2 = [π_ic_1_2, π_ic_2_2, π_ic_3_2, π_ic_4_2]
        
        
        """ ACA FALTA CREAR PENSION CAPITALIZACION COLECTIVA """
        π_cc_1_1 = μ*τ_cc*w*l1[0]*e[0][0] 
        π_cc_2_1 = μ*τ_cc*w*l1[1]*e[0][1] + (1+r)*π_cc_1_1 
        π_cc_3_1 = μ*τ_cc*w*l1[2]*e[0][2] + (1+r)*π_cc_2_1 
        π_cc_4_1 = μ*τ_cc*w*l1[3]*e[0][3] + (1+r)*π_cc_3_1     
        
        π_cc_1_2 = μ*τ_cc*w*l2[0]*e[1][0]
        π_cc_2_2 = μ*τ_cc*w*l2[1]*e[1][1] + (1+r)*π_cc_1_2 
        π_cc_3_2 = μ*τ_cc*w*l2[2]*e[1][2] + (1+r)*π_cc_2_2 
        π_cc_4_2 = μ*τ_cc*w*l2[3]*e[1][3] + (1+r)*π_cc_3_2     
                                    
        
        """ ACA INCLUIR OTRA PRODUCTIVIDAD EN DENOMINADOR """        
        Π_cc_1 = [π_cc_1_1, π_cc_2_1, π_cc_3_1, π_cc_4_1]  
        Π_cc_2 = [π_cc_1_2, π_cc_2_2, π_cc_3_2, π_cc_4_2]  
        
        """ Agencia de pensiones """
        #Individual capitalization 
        ing_ic1 = π_ic_4_1*me1*m[4]
        ing_ic2 = π_ic_4_2*me2*m[4]        
        
        pen_ic_1 = ing_ic1/(me1*(m[4]+m[5]))
        pen_ic_2 = ing_ic2/(me2*(m[4]+m[5]))        
        
        # Collective capitalization 
        pen_cc_2 = self.pen_cc_2 #ahora es valor fijo #(π_cc_4_1 + π_cc_4_2)/(2*TR)
        ing_cc1 = π_cc_4_1*me1*m[4]
        ing_cc2 = π_cc_4_2*me2*m[4]
        
        cost_cc2_4 = pen_cc_2*me2*m[4]
        cost_cc2_5 = pen_cc_2*me2*m[5]
        
        FR_cc = (ing_cc1+ing_cc2) - (cost_cc2_4+cost_cc2_5)
        pen_cc_1 = FR_cc/(me1*(m[4]+m[5]))  
        
        """ Aggregate pension funds """
        #Individual capitalization
        cost_ic1_4 = pen_ic_1*me1*m[4]
        cost_ic1_5 = pen_ic_1*me1*m[5]
        
        cost_ic2_4 = pen_ic_2*me2*m[4]
        cost_ic2_5 = pen_ic_2*me2*m[5]
        
        Π_ic_1_5 = ing_ic1 - cost_ic1_4
        Π_ic_1_6 = Π_ic_1_5 - cost_ic1_5 
        
        Π_ic_2_5 = ing_ic2 - cost_ic2_4
        Π_ic_2_6 = Π_ic_2_5 - cost_ic2_5 
        
        Π_ic_56 = Π_ic_1_5 + Π_ic_2_5 +  Π_ic_1_6 +  Π_ic_2_6 #Π_cc_1_6 +  Π_cc_2_6  =0      
        
        Π_ic = (π_ic_1_1*m[0]*me1 + π_ic_2_1*m[1]*me1 + π_ic_3_1*m[2]*me1 + π_ic_4_1*m[3]*me1) + (π_ic_1_2*m[0]*me2 + π_ic_2_2*m[1]*me2 + π_ic_3_2*m[2]*me2 + π_ic_4_2*m[3]*me2) + (Π_ic_56)

        #Collective capitalization
        cost_cc1_4 = pen_cc_1*me1*m[4]
        cost_cc1_5 = pen_cc_1*me1*m[5]        
        Π_cc_1_5 = ing_cc1 - cost_cc1_4
        Π_cc_1_6 = Π_cc_1_5 - cost_cc1_5 
        
        Π_cc_2_5 = ing_cc2 - cost_cc2_4
        Π_cc_2_6 = Π_cc_2_5 - cost_cc2_5 
        
        Π_cc_56 = Π_cc_1_5 + Π_cc_2_5 +  Π_cc_1_6 +  Π_cc_2_6 #Π_cc_1_6 +  Π_cc_2_6  =0      
          
        Π_cc = (π_cc_1_1*m[0]*me1 + π_cc_2_1*m[1]*me1 + π_cc_3_1*m[2]*me1 + π_cc_4_1*m[3]*me1) + (π_cc_1_2*m[0]*me2 + π_cc_2_2*m[1]*me2 + π_cc_3_2*m[2]*me2 + π_cc_4_2*m[3]*me2) + (Π_cc_56)         
        Πbar = (Π_ic + Π_cc)/Lbar
        
        
        #Pensión solidaria
        Ω1 = ω1[0]*m[1]*ϕ[1]*me1 + ω1[1]*m[2]*ϕ[2]*me1 + ω1[2]*m[3]*ϕ[3]*me1 + ω1[3]*m[4]*ϕ[4]*me1 + ω1[4]*m[5]*ϕ[5]*me1
        Ω2 = ω2[0]*m[1]*ϕ[1]*me2 + ω2[1]*m[2]*ϕ[2]*me2 + ω2[2]*m[3]*ϕ[3]*me2 + ω2[3]*m[4]*ϕ[4]*me2 + ω2[4]*m[5]*ϕ[5]*me2
        
        Ωbar = (Ω1 + Ω2)/Lbar
                
        Tax = τ_w*w*Lbar/(A*Lbar) + τ_r*r*Ω1/Lbar + τ_r*r*Ω2/Lbar + τ_c*Cbar #C ya viene estacionario (por eso no es /L) 
        Beq = ((ω1[0]*m[1]*(1-ϕ[1])*me1 + ω1[1]*m[2]*(1-ϕ[2])*me1 + ω1[2]*m[3]*(1-ϕ[3])*me1 + ω1[3]*m[4]*(1-ϕ[4])*me1 + ω1[4]*m[5]*(1-ϕ[5])*me1) + (ω2[0]*m[1]*(1-ϕ[1])*me2 + ω2[1]*m[2]*(1-ϕ[2])*me2 + ω2[2]*m[3]*(1-ϕ[3])*me2 + ω2[3]*m[4]*(1-ϕ[4])*me2 + ω2[4]*m[5]*(1-ϕ[5])*me2))/Lbar
        tr = ν*(Tax + Beq - g*Kbar**α)*Lbar/(m[4]*me2 + m[5]*me2)        
        pen_sp_1 = 0.
        pen_sp_2 = tr/TR           

        """ AGREGAR PENSION COLECTIVA A CONSUMO """
        #Consumo
        C1 = np.zeros((T+TR))
        C1[0] = ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l1[0]*e[0][0] - ω1[0]*(1+ga))/(1+τ_c)
        C1[1] = ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l1[1]*e[0][1] + (1+(1-τ_r)*r)*ω1[0] - ω1[1]*(1+ga))/(1+τ_c)
        C1[2] = ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l1[2]*e[0][2] + (1+(1-τ_r)*r)*ω1[1] - ω1[2]*(1+ga))/(1+τ_c)
        C1[3] = ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l1[3]*e[0][3] + (1+(1-τ_r)*r)*ω1[2] - ω1[3]*(1+ga))/(1+τ_c)
        C1[4] = (pen_ic_1 + pen_cc_1 + pen + pen_sp_1 + (1+(1-τ_r)*r)*ω1[3] - ω1[4]*(1+ga))/(1+τ_c)
        C1[5] = (pen_ic_1 + pen_cc_1 + pen + pen_sp_1 + (1+(1-τ_r)*r)*ω1[4])/(1+τ_c)               

        C2 = np.zeros((T+TR))
        C2[0] = ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l2[0]*e[1][0] - ω2[0]*(1+ga))/(1+τ_c)
        C2[1] = ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l2[1]*e[1][1] + (1+(1-τ_r)*r)*ω2[0] - ω2[1]*(1+ga))/(1+τ_c)
        C2[2] = ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l2[2]*e[1][2] + (1+(1-τ_r)*r)*ω2[1] - ω2[2]*(1+ga))/(1+τ_c)
        C2[3] = ((1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l2[3]*e[1][3] + (1+(1-τ_r)*r)*ω2[2] - ω2[3]*(1+ga))/(1+τ_c)
        C2[4] = (pen_ic_2 + pen_cc_2 + pen + pen_sp_2 + (1+(1-τ_r)*r)*ω2[3] - ω2[4]*(1+ga))/(1+τ_c)
        C2[5] = (pen_ic_2 + pen_cc_2 + pen + pen_sp_2 + (1+(1-τ_r)*r)*ω2[4])/(1+τ_c)  
        
        if prod==False:
            return ω1, ω2, l1, l2, C1, C2, Π_ic_1, Π_ic_2, Π_cc_1, Π_cc_2, w, r, pen_ic_1, pen_ic_2, pen_cc_1, pen_cc_2, pen , pen_sp_2, Kbar, Lbar, Cbar, Πbar, Ωbar
        else: 
            
            """Aggregate labor"""
            L1 = (e[0][0]*l1[0]*m[0]*me1 + e[0][1]*l1[1]*m[1]*me1 + e[0][2]*l1[2]*m[2]*me1 + e[0][3]*l1[3]*m[3]*me1) 
            L2 = (e[1][0]*l2[0]*m[0]*me2 + e[1][1]*l2[1]*m[1]*me2 + e[1][2]*l2[2]*m[2]*me2 + e[1][3]*l2[3]*m[3]*me2)

            """Aggregate wealth"""
            Ω1 = (ω1[0]*m[1]*ϕ[1]*me1 + ω1[1]*m[2]*ϕ[2]*me1 + ω1[2]*m[3]*ϕ[3]*me1 + ω1[3]*m[4]*ϕ[4]*me1 + ω1[4]*m[5]*ϕ[5]*me1)/Lbar #L1
            Ω2 = (ω2[0]*m[1]*ϕ[1]*me2 + ω2[1]*m[2]*ϕ[2]*me2 + ω2[2]*m[3]*ϕ[3]*me2 + ω2[3]*m[4]*ϕ[4]*me2 + ω2[4]*m[5]*ϕ[5]*me2)/Lbar #L2

            """Aggregate pension funds"""
            #Individual capitalization
            cost_ic1_4 = pen_ic_1*me1*m[4]
            cost_ic1_5 = pen_ic_1*me1*m[5]

            cost_ic2_4 = pen_ic_2*me2*m[4]
            cost_ic2_5 = pen_ic_2*me2*m[5]

            Π_ic_1_5 = ing_ic1 - cost_ic1_4
            Π_ic_1_6 = Π_ic_1_5 - cost_ic1_5 

            Π_ic_2_5 = ing_ic2 - cost_ic2_4
            Π_ic_2_6 = Π_ic_2_5 - cost_ic2_5 

            Π_ic_1 = ((π_ic_1_1*m[0]*me1 + π_ic_2_1*m[1]*me1 + π_ic_3_1*m[2]*me1 + π_ic_4_1*m[3]*me1) + Π_ic_1_5 + Π_ic_1_6)/Lbar #L1
            Π_ic_2 = ((π_ic_1_2*m[0]*me2 + π_ic_2_2*m[1]*me2 + π_ic_3_2*m[2]*me2 + π_ic_4_2*m[3]*me2) + Π_ic_2_5 + Π_ic_2_6)/Lbar #L2

            #Collective capitalization
            cost_cc1_4 = pen_cc_1*me1*m[4]
            cost_cc1_5 = pen_cc_1*me1*m[5]        
            Π_cc_1_5 = ing_cc1 - cost_cc1_4
            Π_cc_1_6 = Π_cc_1_5 - cost_cc1_5 

            Π_cc_2_5 = ing_cc2 - cost_cc2_4
            Π_cc_2_6 = Π_cc_2_5 - cost_cc2_5 

            Π_cc_1 = ((π_cc_1_1*m[0]*me1 + π_cc_2_1*m[1]*me1 + π_cc_3_1*m[2]*me1 + π_cc_4_1*m[3]*me1) + Π_cc_1_5 + Π_cc_1_6)/Lbar #L1
            Π_cc_2 = ((π_cc_1_2*m[0]*me2 + π_cc_2_2*m[1]*me2 + π_cc_3_2*m[2]*me2 + π_cc_4_2*m[3]*me2) + Π_cc_2_5 + Π_cc_2_6)/Lbar #L2

            Π1 = Π_ic_1 + Π_cc_1
            Π2 = Π_ic_2 + Π_cc_2
            
            """Aggregate capital"""
            K1 = (Ω1 + Π_ic_1 + Π_cc_1)
            K2 = (Ω2 + Π_ic_2 + Π_cc_2)
            """Aggregate consumption"""
            CC1 = (C1[0]*m[0]*me1 + C1[1]*m[1]*me1 + C1[2]*m[2]*me1 + C1[3]*m[3]*me1 + C1[4]*m[4]*me1 + C1[5]*m[5]*me1)/Lbar#L1 
            CC2 = (C2[0]*m[0]*me2 + C2[1]*m[1]*me2 + C2[2]*m[2]*me2 + C2[3]*m[3]*me2 + C2[4]*m[4]*me2 + C2[5]*m[5]*me2)/Lbar#L2  
                
            
            return L1, L2, Ω1, Ω2, Π1, Π2, K1, K2, CC1, CC2
    


    def Bequest(self, ω1, ω2, L):
        ϕ, m, τ_w, τ_r, τ_c, A, g, α, me1, me2 = self.ϕ, self.m, self.τ_w, self.τ_r, self.τ_c, self.A, self.g, self.α,  self.me1, self.me2
        ω2_1, ω3_1, ω4_1, ω5_1, ω6_1 = ω1[0], ω1[1], ω1[2], ω1[3], ω1[4]
        ω2_2, ω3_2, ω4_2, ω5_2, ω6_2 = ω2[0], ω2[1], ω2[2], ω2[3], ω2[4]

        """ Estacionario """
        Beq = ((ω2_1*m[1]*(1-ϕ[1])*me1 + ω3_1*m[2]*(1-ϕ[2])*me1 + ω4_1*m[3]*(1-ϕ[3])*me1 + ω5_1*m[4]*(1-ϕ[4])*me1 + ω6_1*m[5]*(1-ϕ[5]))*me1 + (ω2_2*m[1]*(1-ϕ[1])*me2 + ω3_2*m[2]*(1-ϕ[2])*me2 + ω4_2*m[3]*(1-ϕ[3])*me2 + ω5_2*m[4]*(1-ϕ[4])*me2 + ω6_2*m[5]*(1-ϕ[5])*me2))/L

        return Beq
    
  
    #RESIDUAL FUNCTION TRANSITION
    def rftr(self, x, tt, variables, states):
        β, γ, η, T, TR, δ, α, ζ, e, ga, τ_w, g, N, m, ϕ, τ_r, A, τ_c = self.β, self.γ, self.η, self.T, self.TR, self.δ, self.α, self.ζ, self.e, self.ga, self.τ_w, self.g, self.N, self.m, self.ϕ, self.τ_r, self.A, self.τ_c  
        me1, me2 = self.me1, self.me2
        
        μ = self.μ
        
        τicseq, τccseq,  wseq, rseq, pen_pg, τpgseq, ppsseq_2, mseq, pcc2seq = variables[0], variables[1], variables[2], variables[3], variables[4], variables[5], variables[6], variables[7], variables[8]

        coptold_1, loptold_1, ωoptold_1 = states[0], states[1], states[2] 
        coptold_2, loptold_2, ωoptold_2 = states[3], states[4], states[5]
        
        y = np.zeros(18)  
        ω2_1, ω3_1, ω4_1, ω5_1, ω6_1, l1_1, l2_1, l3_1, l4_1 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8]
        ω2_2, ω3_2, ω4_2, ω5_2, ω6_2, l1_2, l2_2, l3_2, l4_2 = x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17]
        
        #Pensions     
        """Individual capitalization (4x2)"""
        #High efficiency
        π_ic_1_1 = μ*τicseq[0]*wseq[0]*l1_1*e[0][0]
        π_ic_2_1 = μ*τicseq[1]*wseq[1]*l2_1*e[0][1] + (1+rseq[1])*π_ic_1_1 
        π_ic_3_1 = μ*τicseq[2]*wseq[2]*l3_1*e[0][2] + (1+rseq[2])*π_ic_2_1 
        π_ic_4_1 = μ*τicseq[3]*wseq[3]*l4_1*e[0][3] + (1+rseq[3])*π_ic_3_1 
        # pen_ic_1 = π_ic_4_1/TR         
        
        π_ic_1_2 = μ*τicseq[0]*wseq[0]*l1_2*e[1][0]
        π_ic_2_2 = μ*τicseq[1]*wseq[1]*l2_2*e[1][1] + (1+rseq[1])*π_ic_1_2 
        π_ic_3_2 = μ*τicseq[2]*wseq[2]*l3_2*e[1][2] + (1+rseq[2])*π_ic_2_2 
        π_ic_4_2 = μ*τicseq[3]*wseq[3]*l4_2*e[1][3] + (1+rseq[3])*π_ic_3_2 
        # pen_ic_2 = π_ic_4_2/TR         
        
        
        """Collective capitalization (4x1)"""
        # π_cc_1 = μ*τccseq[0]*wseq[0]*l1_1*e[0][0] + μ*τccseq[0]*wseq[0]*l1_2*e[1][0] 
        # π_cc_2 = μ*τccseq[1]*wseq[1]*l2_1*e[0][1] + μ*τccseq[1]*wseq[1]*l2_2*e[1][1]  + (1+rseq[1])*π_cc_1
        # π_cc_3 = μ*τccseq[2]*wseq[2]*l3_1*e[0][2] + μ*τccseq[2]*wseq[2]*l3_2*e[1][2]  + (1+rseq[2])*π_cc_2
        # π_cc_4 = μ*τccseq[3]*wseq[3]*l4_1*e[0][3] + μ*τccseq[3]*wseq[3]*l4_2*e[1][3]  + (1+rseq[3])*π_cc_3          
        # pen_cc_1 = π_cc_4/(2*TR) 
        # pen_cc_2 = π_cc_4/(2*TR) 
        
        π_cc_1_1 = μ*τccseq[0]*wseq[0]*l1_1*e[0][0] 
        π_cc_2_1 = μ*τccseq[1]*wseq[1]*l2_1*e[0][1] + (1+rseq[1])*π_cc_1_1
        π_cc_3_1 = μ*τccseq[2]*wseq[2]*l3_1*e[0][2] + (1+rseq[2])*π_cc_2_1
        π_cc_4_1 = μ*τccseq[3]*wseq[3]*l4_1*e[0][3] + (1+rseq[3])*π_cc_3_1        
       
        π_cc_1_2 = μ*τccseq[0]*wseq[0]*l1_2*e[1][0] 
        π_cc_2_2 = μ*τccseq[1]*wseq[1]*l2_2*e[1][1]  + (1+rseq[1])*π_cc_1_2
        π_cc_3_2 = μ*τccseq[2]*wseq[2]*l3_2*e[1][2]  + (1+rseq[2])*π_cc_2_2
        π_cc_4_2 = μ*τccseq[3]*wseq[3]*l4_2*e[1][3]  + (1+rseq[3])*π_cc_3_2          
        
        
        """ Agencia de pensiones """
        #Individual capitalization 
        ing_ic1 = π_ic_4_1*me1*mseq[4]
        ing_ic2 = π_ic_4_2*me2*mseq[4]        
        
        pen_ic_1 = ing_ic1/(me1*(mseq[4]+mseq[5]))
        pen_ic_2 = ing_ic2/(me2*(mseq[4]+mseq[5]))        
        
        # Collective capitalization 
        # pen_cc_2 = self.pen_cc_2 #ahora es valor fijo #(π_cc_4_1 + π_cc_4_2)/(2*TR)
        ing_cc1 = π_cc_4_1*me1*mseq[4]
        ing_cc2 = π_cc_4_2*me2*mseq[4]
        
        cost_cc2_4 = pcc2seq[4]*me2*mseq[4] #pen_cc_2*me2*mseq[4]
        cost_cc2_5 = pcc2seq[5]*me2*mseq[5] #pen_cc_2*me2*mseq[5]
        
        FR_cc = (ing_cc1+ing_cc2) - (cost_cc2_4+cost_cc2_5)
        pen_cc_1 = FR_cc/(me1*(mseq[4]+mseq[5]))  #(π_cc_4_1 + π_cc_4_2)/(2*TR)  
        
        
        """ EN MODELO CON DOS PRODUCTIVIDADES EN CONSUMO PENSIONADOS AGREGAR ppsseq_2[4] y ppsseq_2[5] """
        
        
        #Consumption
        c1_1 = ((1-τ_w-μ*τicseq[0]-μ*τccseq[0]-(1-μ)*τpgseq[0])*wseq[0]*l1_1*e[0][0] - ω2_1*(1+ga))/(1+τ_c)
        c2_1 = ((1-τ_w-μ*τicseq[1]-μ*τccseq[1]-(1-μ)*τpgseq[1])*wseq[1]*l2_1*e[0][1] + (1+(1-τ_r)*rseq[1])*ω2_1 - ω3_1*(1+ga))/(1+τ_c)
        c3_1 = ((1-τ_w-μ*τicseq[2]-μ*τccseq[2]-(1-μ)*τpgseq[2])*wseq[2]*l3_1*e[0][2] + (1+(1-τ_r)*rseq[2])*ω3_1 - ω4_1*(1+ga))/(1+τ_c)
        c4_1 = ((1-τ_w-μ*τicseq[3]-μ*τccseq[3]-(1-μ)*τpgseq[3])*wseq[3]*l4_1*e[0][3] + (1+(1-τ_r)*rseq[3])*ω4_1 - ω5_1*(1+ga))/(1+τ_c)
        c5_1 = (pen_ic_1 + pen_cc_1 + pen_pg[4] + 0.0 + (1+(1-τ_r)*rseq[4])*ω5_1 - ω6_1*(1+ga))/(1+τ_c) 
        c6_1 = (pen_ic_1 + pen_cc_1 + pen_pg[5] + 0.0 + (1+(1-τ_r)*rseq[5])*ω6_1)/(1+τ_c)
                
        c1_2 = ((1-τ_w-μ*τicseq[0]-μ*τccseq[0]-(1-μ)*τpgseq[0])*wseq[0]*l1_2*e[1][0] - ω2_2*(1+ga))/(1+τ_c)
        c2_2 = ((1-τ_w-μ*τicseq[1]-μ*τccseq[1]-(1-μ)*τpgseq[1])*wseq[1]*l2_2*e[1][1] + (1+(1-τ_r)*rseq[1])*ω2_2 - ω3_2*(1+ga))/(1+τ_c)
        c3_2 = ((1-τ_w-μ*τicseq[2]-μ*τccseq[2]-(1-μ)*τpgseq[2])*wseq[2]*l3_2*e[1][2] + (1+(1-τ_r)*rseq[2])*ω3_2 - ω4_2*(1+ga))/(1+τ_c)
        c4_2 = ((1-τ_w-μ*τicseq[3]-μ*τccseq[3]-(1-μ)*τpgseq[3])*wseq[3]*l4_2*e[1][3] + (1+(1-τ_r)*rseq[3])*ω4_2 - ω5_2*(1+ga))/(1+τ_c)
        c5_2 = (pen_ic_2 + pcc2seq[4] + pen_pg[4] + ppsseq_2[4] + (1+(1-τ_r)*rseq[4])*ω5_2 - ω6_2*(1+ga))/(1+τ_c) 
        c6_2 = (pen_ic_2 + pcc2seq[5] + pen_pg[5] + ppsseq_2[5] + (1+(1-τ_r)*rseq[5])*ω6_2)/(1+τ_c)
          
        #Labor supply
        y[0] = self.ul(c1_1, 1-l1_1)/self.uc(c1_1, 1-l1_1) - ((1-τ_w-μ*τicseq[0]-μ*τccseq[0]-(1-μ)*τpgseq[0])*wseq[0]*e[0][0])/(1+τ_c)
        y[1] = self.ul(c2_1, 1-l2_1)/self.uc(c2_1, 1-l2_1) - ((1-τ_w-μ*τicseq[1]-μ*τccseq[1]-(1-μ)*τpgseq[1])*wseq[1]*e[0][1])/(1+τ_c)
        y[2] = self.ul(c3_1, 1-l3_1)/self.uc(c3_1, 1-l3_1) - ((1-τ_w-μ*τicseq[2]-μ*τccseq[2]-(1-μ)*τpgseq[2])*wseq[2]*e[0][2])/(1+τ_c)
        y[3] = self.ul(c4_1, 1-l4_1)/self.uc(c4_1, 1-l4_1) - ((1-τ_w-μ*τicseq[3]-μ*τccseq[3]-(1-μ)*τpgseq[3])*wseq[3]*e[0][3])/(1+τ_c)

        y[4] = self.ul(c1_2, 1-l1_2)/self.uc(c1_2, 1-l1_2) - ((1-τ_w-μ*τicseq[0]-μ*τccseq[0]-(1-μ)*τpgseq[0])*wseq[0]*e[1][0])/(1+τ_c)
        y[5] = self.ul(c2_2, 1-l2_2)/self.uc(c2_2, 1-l2_2) - ((1-τ_w-μ*τicseq[1]-μ*τccseq[1]-(1-μ)*τpgseq[1])*wseq[1]*e[1][1])/(1+τ_c)
        y[6] = self.ul(c3_2, 1-l3_2)/self.uc(c3_2, 1-l3_2) - ((1-τ_w-μ*τicseq[2]-μ*τccseq[2]-(1-μ)*τpgseq[2])*wseq[2]*e[1][2])/(1+τ_c)
        y[7] = self.ul(c4_2, 1-l4_2)/self.uc(c4_2, 1-l4_2) - ((1-τ_w-μ*τicseq[3]-μ*τccseq[3]-(1-μ)*τpgseq[3])*wseq[3]*e[1][3])/(1+τ_c)

        #Euler equation
        y[8:18] = x[8:18]

#         #in period 1: unexpected change, therefore, prior to period 1, all agents
#          #behave as in the old steady state
        if tt <= 0:
            c1_1 = coptold_1[0]
            c1_2 = coptold_2[0]
            y[0] = l1_1 - loptold_1[0]
            y[4] = l1_2 - loptold_2[0]
            y[8] = ω2_1 - ωoptold_1[0]
            y[13] = ω2_2 - ωoptold_2[0]
        else:
            y[8] = (1+ga)*ϕ[0]/((β*(1+(1-τ_r)*rseq[1]))*ϕ[1]) - self.uc(c2_1,1-l2_1)/self.uc(c1_1,1-l1_1) 
            y[13] = (1+ga)*ϕ[0]/((β*(1+(1-τ_r)*rseq[1]))*ϕ[1]) - self.uc(c2_2,1-l2_2)/self.uc(c1_2,1-l1_2)             
            
        if tt <= -1:
            c2_1 = coptold_1[1]
            c2_2 = coptold_2[1]
            y[1] = l2_1 - loptold_1[1]
            y[5] = l2_2 - loptold_2[1]
            y[9] = ω3_1 - ωoptold_1[1]
            y[14] = ω3_2 - ωoptold_2[1]            
        else:
            y[9]  = (1+ga)*ϕ[0]/((β*(1+(1-τ_r)*rseq[2]))*ϕ[2]) - self.uc(c3_1,1-l3_1)/self.uc(c2_1,1-l2_1) 
            y[14] = (1+ga)*ϕ[0]/((β*(1+(1-τ_r)*rseq[2]))*ϕ[2]) - self.uc(c3_2,1-l3_2)/self.uc(c2_2,1-l2_2) 
            
            
        if tt <= -2:
            c3_1 = coptold_1[2]
            c3_2 = coptold_2[2]
            y[2] = l3_1 - loptold_1[2]
            y[6] = l3_2 - loptold_2[2]
            y[10] = ω4_1 - ωoptold_1[2]
            y[15] = ω4_2 - ωoptold_2[2]            
            
            
        else: 
            y[10] = (1+ga)*ϕ[0]/((β*(1+(1-τ_r)*rseq[3]))*ϕ[3]) - self.uc(c4_1,1-l4_1)/self.uc(c3_1,1-l3_1) 
            y[15] = (1+ga)*ϕ[0]/((β*(1+(1-τ_r)*rseq[3]))*ϕ[3]) - self.uc(c4_2,1-l4_2)/self.uc(c3_2,1-l3_2) 
             
        if tt <= -3:
            c4_1 = coptold_1[3]
            c4_2 = coptold_2[3]
            y[3] = l4_1 - loptold_1[3]
            y[7] = l4_2 - loptold_2[3]
            y[11] = ω5_1 - ωoptold_1[3]
            y[16] = ω5_2 - ωoptold_2[3]
            
        else:
            y[11] = (1+ga)*ϕ[0]/((β*(1+(1-τ_r)*rseq[4]))*ϕ[4]) - self.uc(c5_1,1)/self.uc(c4_1,1-l4_1) 
            y[16] = (1+ga)*ϕ[0]/((β*(1+(1-τ_r)*rseq[4]))*ϕ[4]) - self.uc(c5_2,1)/self.uc(c4_2,1-l4_2) 
             
        if tt <= -4:
            c5_1 = coptold_1[4]
            c5_2 = coptold_2[4]
            y[12] = ω6_1 - ωoptold_1[4]
            y[17] = ω6_2 - ωoptold_2[4]
            
        else: 
            y[12] = (1+ga)*ϕ[0]/((β*(1+(1-τ_r)*rseq[5]))*ϕ[5]) - self.uc(c6_1,1)/self.uc(c5_1,1)
            y[17] = (1+ga)*ϕ[0]/((β*(1+(1-τ_r)*rseq[5]))*ϕ[5]) - self.uc(c6_2,1)/self.uc(c5_2,1)            
        return y         
    
   
    def transition(self, update_t, update_new, update_old, ωlopt, states):
        nt, T, TR, e, τ_w, τ_r, τ_c, ga, g, A, m, ϕ, mt, m0, m1 = self.nt, self.T, self.TR, self.e, self.τ_w, self.τ_r, self.τ_c, self.ga, self.g, self.A, self.m, self.ϕ, self.mt, self.m0, self.m1
        me1, me2 = self.me1, self.me2
        
        μ = self.μ
        """ Initialization """
        #Precios
        wseq = np.zeros(T+TR) 
        rseq = np.zeros(T+TR)
        
        # picseq_1, picseq_2 = np.zeros(T+TR), np.zeros(T+TR) #Pensión capitalización individual
        # pccseq_1, pccseq_2 = np.zeros(T+TR), np.zeros(T+TR) #Pensión capitalización colectiva
        
        #Pensiones
        ppgseq_1 = np.zeros(T+TR) #Pensión pay as you go. Igual en ambas productividades
        ppsseq_2 = np.zeros(T+TR) #Pensión solidaria. Es cero en productividad alta
        τicseq, τccseq, τpgseq  = np.zeros(T+TR), np.zeros(T+TR), np.zeros(T+TR)
        pcc2seq = np.zeros(T+TR)
        #Demografía
        mseq = np.zeros(T+TR) 
        
        #Variables agregadas
        Ωtnew, Ωtlnew, ltnew, πictnew, πicltnew, πcctnew, πccltnew, ctnew, ctlnew, ktnew, Beqtnew, Beqltnew = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
        
        ltnew_1, ltnew_2, Ωtnew_1, Ωtnew_2, πictnew_1, πictnew_2, πcctnew_1, πcctnew_2, ktnew_1, ktnew_2, ctnew_1, ctnew_2 = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
        
        ltlnew_1, ltlnew_2, Ωtlnew_1, Ωtlnew_2, πtlnew_1, πtlnew_2, ktlnew_1, ktlnew_2, ctlnew_1, ctlnew_2 = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
        

        #Variables individuales en el momento t
        copt_1  = np.zeros((nt+5, T+TR))
        loptt_1 = np.zeros((nt+5, T+TR))   
        copt_2  = np.zeros((nt+5, T+TR))
        loptt_2 = np.zeros((nt+5, T+TR))           
        mopt    = np.zeros((nt+5, T+TR))   
        
        #ωlopt (ω1_1, l1_1, ω2_1, l2_1)
        ωopt_1 = np.concatenate([[0],  ωlopt[0:5]])
        lopt_1 = ωlopt[5:9]
        ωopt_2 = np.concatenate([[0],  ωlopt[9:14]])
        lopt_2 = ωlopt[14:18]
        
        """ ACA: AGREGAR PENSION SOLIDARIA """
        τict, τcct, wt, rt, ppgt, τpgt, pps2t, pen_cc_2_t = update_t[0], update_t[1], update_t[2], update_t[3], update_t[4], update_t[5], update_t[6], update_t[7]
        τicnew, τccnew, wnew, rnew, ppgnew, τpgnew, pps2new, pen_cc_2_new = update_new[0], update_new[1], update_new[2], update_new[3], update_new[4], update_new[5], update_new[6], update_new[7]
        τicold, τccold, wold, rold, ppgold, τpgold, pps2old, pen_cc_2_old = update_old[0], update_old[1], update_old[2], update_old[3], update_old[4], update_old[5], update_old[6], update_old[7]
    
    
        #Sequence prices
        tt = nt+1
        # print("tt:", tt)
        while tt != -4:
            tt = tt-1
            self.A = self.At[tt-1]
            A = self.A
            # self.m = self.mt[tt-1]
            # self.ϕ = self.ϕt[tt-1]
            
            # m = self.m
            # ϕ = self.ϕ
            
            
            #wage, interest rate and pensions over the lifetime of the individual
            if tt > nt-5: #Include new steady state prices
                wseq[0:nt-tt+1 ]  = wt[tt-1:nt]
                rseq[0:nt-tt+1]   = rt[tt-1:nt]  
                ppgseq_1[0:nt-tt+1] = ppgt[tt-1:nt]                  
                τicseq[0:nt-tt+1]   = τict[tt-1:nt]     
                τccseq[0:nt-tt+1]   = τcct[tt-1:nt]   
                τpgseq[0:nt-tt+1]   = τpgt[tt-1:nt] 
                ppsseq_2[0:nt-tt+1] = pps2t[tt-1:nt]  
                pcc2seq[0:nt-tt+1] = pen_cc_2_t[tt-1:nt]  
                for i_m in range(nt-tt+1):
                    mseq[i_m] = mt[tt+i_m-1][i_m]
                
                wseq[nt-tt+2-1:6]    = np.ones(6-nt+tt-1)*wnew
                rseq[nt-tt+2 -1:6]   = np.ones(6-nt+tt-1)*rnew
                ppgseq_1[nt-tt+2 -1:6] = np.ones(6-nt+tt-1)*ppgnew                
                τicseq[nt-tt+2 -1:6]   = np.ones(6-nt+tt-1)*τicnew
                τccseq[nt-tt+2 -1:6]   = np.ones(6-nt+tt-1)*τccnew
                τpgseq[nt-tt+2 -1:6]   = np.ones(6-nt+tt-1)*τpgnew
                ppsseq_2[nt-tt+2 -1:6] = np.ones(6-nt+tt-1)*pps2new  
                pcc2seq[nt-tt+2 -1:6] = np.ones(6-nt+tt-1)*pen_cc_2_new  
                mseq[nt-tt+2-1:6]    = m1[nt-tt+2-1:6] 

            elif tt<1: #Include old steady state prices
                wseq[0:1-tt] = np.ones(-tt+1)*wold 
                rseq[0:1-tt] = np.ones(-tt+1)*rold
                ppgseq_1[0:1-tt] = np.ones(-tt+1)*ppgold                  
                τicseq[0:1-tt]=np.ones(-tt+1)*τicold        
                τccseq[0:1-tt]=np.ones(-tt+1)*τccold
                τpgseq[0:1-tt]=np.ones(-tt+1)*τpgold
                ppsseq_2[0:1-tt] = np.ones(-tt+1)*pps2old   
                pcc2seq[0:1-tt] = np.ones(-tt+1)*pen_cc_2_old
                mseq[0:1-tt] = m0[0:1-tt] 
                
                wseq[2-tt -1:6] = wt[0:4+tt+1]
                rseq[2-tt -1:6] = rt[0:4+tt+1]
                ppgseq_1[2-tt -1:6] = ppgt[0:4+tt+1]                
                τicseq[2-tt -1:6] = τict[0:4+tt+1]
                τccseq[2-tt -1:6] = τcct[0:4+tt+1]
                τpgseq[2-tt -1:6] = τpgt[0:4+tt+1]
                ppsseq_2[2-tt -1:6] = pps2t[0:4+tt+1]   
                pcc2seq[2-tt -1:6] = pen_cc_2_t[0:4+tt+1]   
                for i_m in range(2-tt -1, 6):
                    mseq[i_m] = mt[i_m+tt-1][i_m]                
                
            else:  #Only transition prices
                wseq[0:6] = wt[tt -1:tt+5]
                rseq[0:6] = rt[tt -1:tt+5]
                ppgseq_1[0:6] = ppgt[tt -1:tt+5]
                τicseq[0:6] = τict[tt -1:tt+5] 
                τccseq[0:6] = τcct[tt -1:tt+5] 
                τpgseq[0:6] = τpgt[tt -1:tt+5] 
                ppsseq_2[0:6] = pps2t[tt -1:tt+5]
                pcc2seq[0:6] = pen_cc_2_t[tt -1:tt+5]
                for i_m in range(6):
                    mseq[i_m] = mt[tt+i_m-1][i_m]
                

            #Solve system of equation
            x0 = np.concatenate([ωopt_1[1:6], lopt_1, ωopt_2[1:6], lopt_2])
            """ ACA: AGREGAR PENSION SOLIDARIA """
            variables = [τicseq, τccseq, wseq, rseq, ppgseq_1, τpgseq, ppsseq_2, mseq, pcc2seq]
            sol = fsolve(self.rftr, x0, args=(tt, variables, states))
#             sol = self.rftr(x0, tt, variables, states)

            ωopt_1[0]   = 0
            ωopt_1[1:6] = sol[0:5]
            lopt_1      = sol[5:9]  
            ωopt_2[0]   = 0
            ωopt_2[1:6] = sol[9:14]
            lopt_2      = sol[14:18]              
            # print("tt:", tt)
            # print("ωopt_2:", ωopt_2)

            #Pensions
            """Individual capitalization (4x2)"""
            #High efficiency
            π_ic_1_1 = μ*τicseq[0]*wseq[0]*lopt_1[0]*e[0][0]
            π_ic_2_1 = μ*τicseq[1]*wseq[1]*lopt_1[1]*e[0][1] + (1+rseq[1])*π_ic_1_1
            π_ic_3_1 = μ*τicseq[2]*wseq[2]*lopt_1[2]*e[0][2] + (1+rseq[2])*π_ic_2_1
            π_ic_4_1 = μ*τicseq[3]*wseq[3]*lopt_1[3]*e[0][3] + (1+rseq[3])*π_ic_3_1        

             #Low efficiency
            π_ic_1_2 = μ*τicseq[0]*wseq[0]*lopt_2[0]*e[1][0]
            π_ic_2_2 = μ*τicseq[1]*wseq[1]*lopt_2[1]*e[1][1] + (1+rseq[1])*π_ic_1_2
            π_ic_3_2 = μ*τicseq[2]*wseq[2]*lopt_2[2]*e[1][2] + (1+rseq[2])*π_ic_2_2
            π_ic_4_2 = μ*τicseq[3]*wseq[3]*lopt_2[3]*e[1][3] + (1+rseq[3])*π_ic_3_2   
                        
            """ ACA: INCLUIR SeGUNDA PRODUCTIVIDAD """
            """Collective capitalization (4x1)"""
            π_cc_1_1 = μ*τccseq[0]*wseq[0]*lopt_1[0]*e[0][0] 
            π_cc_2_1 = μ*τccseq[1]*wseq[1]*lopt_1[1]*e[0][1] + (1+rseq[1])*π_cc_1_1
            π_cc_3_1 = μ*τccseq[2]*wseq[2]*lopt_1[2]*e[0][2] + (1+rseq[2])*π_cc_2_1
            π_cc_4_1 = μ*τccseq[3]*wseq[3]*lopt_1[3]*e[0][3] + (1+rseq[3])*π_cc_3_1
          
            π_cc_1_2 = μ*τccseq[0]*wseq[0]*lopt_2[0]*e[1][0]
            π_cc_2_2 = μ*τccseq[1]*wseq[1]*lopt_2[1]*e[1][1] + (1+rseq[1])*π_cc_1_2
            π_cc_3_2 = μ*τccseq[2]*wseq[2]*lopt_2[2]*e[1][2] + (1+rseq[2])*π_cc_2_2
            π_cc_4_2 = μ*τccseq[3]*wseq[3]*lopt_2[3]*e[1][3] + (1+rseq[3])*π_cc_3_2 
                  
            """ Agencia de pensiones """
            #Individual capitalization 
            ing_ic1 = π_ic_4_1*me1*mseq[4]
            ing_ic2 = π_ic_4_2*me2*mseq[4]        

            pen_ic_1 = ing_ic1/(me1*(mseq[4]+mseq[5]))
            pen_ic_2 = ing_ic2/(me2*(mseq[4]+mseq[5]))        

            # Collective capitalization 
            # pen_cc_2 = self.pen_cc_2 #ahora es valor fijo #(π_cc_4_1 + π_cc_4_2)/(2*TR)
            ing_cc1 = π_cc_4_1*me1*mseq[4]
            ing_cc2 = π_cc_4_2*me2*mseq[4]

            cost_cc2_4 = pcc2seq[4]*me2*mseq[4] #pen_cc_2*me2*mseq[4]
            cost_cc2_5 = pcc2seq[5]*me2*mseq[5] #pen_cc_2*me2*mseq[5]

            FR_cc = (ing_cc1+ing_cc2) - (cost_cc2_4+cost_cc2_5)
            pen_cc_1 = FR_cc/(me1*(mseq[4]+mseq[5]))  #(π_cc_4_1 + π_cc_4_2)/(2*TR)  
          
            """Aggregate pension funds"""
            #Individual capitalization
            cost_ic1_4 = pen_ic_1*me1*mseq[4]
            cost_ic1_5 = pen_ic_1*me1*mseq[5]

            cost_ic2_4 = pen_ic_2*me2*mseq[4]
            cost_ic2_5 = pen_ic_2*me2*mseq[5]

            Π_ic_1_5 = ing_ic1 - cost_ic1_4
            Π_ic_1_6 = Π_ic_1_5 - cost_ic1_5 

            Π_ic_2_5 = ing_ic2 - cost_ic2_4
            Π_ic_2_6 = Π_ic_2_5 - cost_ic2_5 
    
            Π_ic_1 = np.array([π_ic_1_1, π_ic_2_1, π_ic_3_1, π_ic_4_1, Π_ic_1_5, Π_ic_1_6])  #Π_ic_1_5, Π_ic_1_6 incluyen m y me
            Π_ic_2 = np.array([π_ic_1_2, π_ic_2_2, π_ic_3_2, π_ic_4_2, Π_ic_2_5, Π_ic_2_6])  #Π_ic_2_5, Π_ic_2_6 incluyen m y me     

            #Collective capitalization
            cost_cc1_4 = pen_cc_1*me1*mseq[4]
            cost_cc1_5 = pen_cc_1*me1*mseq[5]       
            Π_cc_1_5 = ing_cc1 - cost_cc1_4
            Π_cc_1_6 = Π_cc_1_5 - cost_cc1_5 

            Π_cc_2_5 = ing_cc2 - cost_cc2_4
            Π_cc_2_6 = Π_cc_2_5 - cost_cc2_5 
            
            Π_cc_1 = np.array( [π_cc_1_1, π_cc_2_1, π_cc_3_1, π_cc_4_1, Π_cc_1_5, Π_cc_1_6] )   #Π_cc_1_5, Π_cc_1_6 incluyen m y me        
            Π_cc_2 = np.array( [π_cc_1_2, π_cc_2_2, π_cc_3_2, π_cc_4_2, Π_cc_2_5, Π_cc_2_6] )   #Π_cc_2_5, Π_cc_2_6 incluyen m y me          

            
            """ ACA: AGREGAR PENSION SOLIDARIA  EN EFICIENCIA 2 """
            #Get individual consumption       
            copt_1[tt+4, 0] = ((1-τ_w-μ*τicseq[0]-μ*τccseq[0]-(1-μ)*τpgseq[0])*wseq[0]*lopt_1[0]*e[0][0]                                 - ωopt_1[1]*(1+ga))/(1+τ_c)
            copt_1[tt+4, 1] = ((1-τ_w-μ*τicseq[1]-μ*τccseq[1]-(1-μ)*τpgseq[1])*wseq[1]*lopt_1[1]*e[0][1] + (1+(1-τ_r)*rseq[1])*ωopt_1[1] - ωopt_1[2]*(1+ga))/(1+τ_c)
            copt_1[tt+4, 2] = ((1-τ_w-μ*τicseq[2]-μ*τccseq[2]-(1-μ)*τpgseq[2])*wseq[2]*lopt_1[2]*e[0][2] + (1+(1-τ_r)*rseq[2])*ωopt_1[2] - ωopt_1[3]*(1+ga))/(1+τ_c)
            copt_1[tt+4, 3] = ((1-τ_w-μ*τicseq[3]-μ*τccseq[3]-(1-μ)*τpgseq[3])*wseq[3]*lopt_1[3]*e[0][3] + (1+(1-τ_r)*rseq[3])*ωopt_1[3] - ωopt_1[4]*(1+ga))/(1+τ_c)
            copt_1[tt+4, 4] = (pen_ic_1 + pen_cc_1 + ppgseq_1[4]                                       + (1+(1-τ_r)*rseq[4])*ωopt_1[4] - ωopt_1[5]*(1+ga))/(1+τ_c)
            copt_1[tt+4, 5] = (pen_ic_1 + pen_cc_1 + ppgseq_1[5]                                       + (1+(1-τ_r)*rseq[5])*ωopt_1[5])/(1+τ_c)
            
            copt_2[tt+4, 0] = ((1-τ_w-μ*τicseq[0]-μ*τccseq[0]-(1-μ)*τpgseq[0])*wseq[0]*lopt_2[0]*e[1][0]                                 - ωopt_2[1]*(1+ga))/(1+τ_c)
            copt_2[tt+4, 1] = ((1-τ_w-μ*τicseq[1]-μ*τccseq[1]-(1-μ)*τpgseq[1])*wseq[1]*lopt_2[1]*e[1][1] + (1+(1-τ_r)*rseq[1])*ωopt_2[1] - ωopt_2[2]*(1+ga))/(1+τ_c)
            copt_2[tt+4, 2] = ((1-τ_w-μ*τicseq[2]-μ*τccseq[2]-(1-μ)*τpgseq[2])*wseq[2]*lopt_2[2]*e[1][2] + (1+(1-τ_r)*rseq[2])*ωopt_2[2] - ωopt_2[3]*(1+ga))/(1+τ_c)
            copt_2[tt+4, 3] = ((1-τ_w-μ*τicseq[3]-μ*τccseq[3]-(1-μ)*τpgseq[3])*wseq[3]*lopt_2[3]*e[1][3] + (1+(1-τ_r)*rseq[3])*ωopt_2[3] - ωopt_2[4]*(1+ga))/(1+τ_c)
            copt_2[tt+4, 4] = (pen_ic_2 + pcc2seq[4] + ppgseq_1[4] +  ppsseq_2[4]                          + (1+(1-τ_r)*rseq[4])*ωopt_2[4] - ωopt_2[5]*(1+ga))/(1+τ_c)
            copt_2[tt+4, 5] = (pen_ic_2 + pcc2seq[5] + ppgseq_1[5] +  ppsseq_2[5]                          + (1+(1-τ_r)*rseq[5])*ωopt_2[5])/(1+τ_c) #ppsseq_1 es igual en ambos casos
                        
            #Estructura demográfica
            for i in range(T+TR):
                mopt[tt+4, i] = mseq[i]

            
            #Get individual labor
            loptt_1[tt+4, 0] = lopt_1[0]  
            loptt_1[tt+4, 1] = lopt_1[1] 
            loptt_1[tt+4, 2] = lopt_1[2]
            loptt_1[tt+4, 3] = lopt_1[3] 
            loptt_1[tt+4, 4] = 0
            loptt_1[tt+4, 5] = 0    
            
            loptt_2[tt+4, 0] = lopt_2[0]  
            loptt_2[tt+4, 1] = lopt_2[1] 
            loptt_2[tt+4, 2] = lopt_2[2]
            loptt_2[tt+4, 3] = lopt_2[3] 
            loptt_2[tt+4, 4] = 0
            loptt_2[tt+4, 5] = 0    
            
            #Get aggregate capital and labor (horizontaly)
            tp = tt-1
            i = 0
            while tp!=nt and i!=6:
                
                tp = tp+1
                i = i+1 
                if tp>0: #Actualizar la agregación del capital y empleo
                    """Aggregate consumption"""
                    ctnew[tp-1] = ctnew[tp-1] + (copt_1[tt+4, i-1]*mopt[tt+4, i-1]*me1) + (copt_2[tt+4, i-1]*mopt[tt+4, i-1]*me2)
                    ctnew_1[tp-1] = ctnew_1[tp-1] + (copt_1[tt+4, i-1]*mopt[tt+4, i-1]*me1)
                    ctnew_2[tp-1] = ctnew_2[tp-1] + (copt_2[tt+4, i-1]*mopt[tt+4, i-1]*me2)
                        
                    """Aggregate savings"""               
                    Ωtnew[tp-1] = Ωtnew[tp-1] + (ωopt_1[i-1]*mopt[tt+4, i-1]*ϕ[i-1]*me1) + (ωopt_2[i-1]*mopt[tt+4, i-1]*ϕ[i-1]*me2) 
                    Ωtnew_1[tp-1] = Ωtnew_1[tp-1] + (ωopt_1[i-1]*mopt[tt+4, i-1]*ϕ[i-1]*me1) 
                    Ωtnew_2[tp-1] = Ωtnew_2[tp-1] + (ωopt_2[i-1]*mopt[tt+4, i-1]*ϕ[i-1]*me2) 
                    
                    
                    """ ACA: AGREGAR BEQUEST"""
                    Beqtnew[tp-1] = Beqtnew[tp-1] + (ωopt_1[i-1]*mopt[tt+4, i-1]*(1-ϕ[i-1])*me1) + (ωopt_2[i-1]*mopt[tt+4, i-1]*(1-ϕ[i-1])*me2) 

                    if i<= 4: 
                        """Labor supply"""                         
                        # ltnew[tp-1] = ltnew[tp-1] + (lopt_1[i-1]*m[i-1]*e[0][i-1])  
                        ltnew[tp-1] = ltnew[tp-1] + (lopt_1[i-1]*mopt[tt+4, i-1]*e[0][i-1]*me1)  + (lopt_2[i-1]*mopt[tt+4, i-1]*e[1][i-1]*me2)  
                        ltnew_1[tp-1] = ltnew_1[tp-1] + (lopt_1[i-1]*mopt[tt+4, i-1]*e[0][i-1]*me1)  
                        ltnew_2[tp-1] = ltnew_2[tp-1] + (lopt_2[i-1]*mopt[tt+4, i-1]*e[1][i-1]*me2)  
                        
                        """Aggregate pension funds"""                    
                        πictnew[tp-1] = πictnew[tp-1] + Π_ic_1[i-1]*mopt[tt+4, i-1]*me1 + Π_ic_2[i-1]*mopt[tt+4, i-1]*me2 
                        πcctnew[tp-1] = πcctnew[tp-1] + Π_cc_1[i-1]*mopt[tt+4, i-1]*me1 + Π_cc_2[i-1]*mopt[tt+4, i-1]*me2  
                        
                        πictnew_1[tp-1] = πictnew_1[tp-1] + Π_ic_1[i-1]*mopt[tt+4, i-1]*me1 
                        πcctnew_1[tp-1] = πcctnew_1[tp-1] + Π_cc_1[i-1]*mopt[tt+4, i-1]*me1  
                        
                        πictnew_2[tp-1] = πictnew_2[tp-1] + Π_ic_2[i-1]*mopt[tt+4, i-1]*me2 
                        πcctnew_2[tp-1] = πcctnew_2[tp-1] + Π_cc_2[i-1]*mopt[tt+4, i-1]*me2                          
                    else:
                        πictnew[tp-1] = πictnew[tp-1] + Π_ic_1[i-1] + Π_ic_2[i-1] #Incluye m y me1 y me2
                        πcctnew[tp-1] = πcctnew[tp-1] + Π_cc_1[i-1] + Π_cc_2[i-1] #Incluye m y me1 y me2                            

                        πictnew_1[tp-1] = πictnew_1[tp-1] + Π_ic_1[i-1] #Incluye m y me1 y me2
                        πcctnew_1[tp-1] = πcctnew_1[tp-1] + Π_cc_1[i-1] #Incluye m y me1 y me2       
                        
                        πictnew_2[tp-1] = πictnew_2[tp-1] + Π_ic_2[i-1] #Incluye m y me1 y me2
                        πcctnew_2[tp-1] = πcctnew_2[tp-1] + Π_cc_2[i-1] #Incluye m y me1 y me2                               


            """Aggregate capital"""
            tp = tt-1
            while tp!=nt:
                tp = tp+1
                if tp>0:     
                    # ktnew[tp-1] = (Ωtnew[tp-1] )/ltnew[tp-1]
                    ktnew[tp-1] = (Ωtnew[tp-1] + πictnew[tp-1] + πcctnew[tp-1])/ltnew[tp-1]
                    Ωtlnew[tp-1] = Ωtnew[tp-1]/ltnew[tp-1]
                    ctlnew[tp-1] = (ctnew[tp-1] )/ltnew[tp-1]
                    πicltnew[tp-1] = (πictnew[tp-1] )/ltnew[tp-1]
                    πccltnew[tp-1] = (πcctnew[tp-1] )/ltnew[tp-1]
                    Beqltnew[tp-1] = (Beqtnew[tp-1] )/ltnew[tp-1]
                    
                    #Variables por productividad
                    ktlnew_1[tp-1] = (Ωtnew_1[tp-1] + πictnew_1[tp-1] + πcctnew_1[tp-1])/ltnew[tp-1] #ltnew_1[tp-1]
                    ktlnew_2[tp-1] = (Ωtnew_2[tp-1] + πictnew_2[tp-1] + πcctnew_2[tp-1])/ltnew[tp-1] #ltnew_2[tp-1]

                    Ωtlnew_1[tp-1] = Ωtnew_1[tp-1]/ltnew[tp-1] #ltnew_1[tp-1]
                    Ωtlnew_2[tp-1] = Ωtnew_2[tp-1]/ltnew[tp-1] #ltnew_2[tp-1]
                    
                    πtlnew_1[tp-1] = (πictnew_1[tp-1] + πcctnew_1[tp-1])/ltnew[tp-1] #ltnew_1[tp-1]
                    πtlnew_2[tp-1] = (πictnew_2[tp-1] + πcctnew_2[tp-1])/ltnew[tp-1] #ltnew_2[tp-1]
                    
                    ctlnew_1[tp-1] = (ctnew_1[tp-1] )/ltnew[tp-1] #ltnew_1[tp-1]
                    ctlnew_2[tp-1] = (ctnew_2[tp-1] )/ltnew[tp-1] #ltnew_2[tp-1]                    
                      

        # if prod==False:
        return Ωtlnew, ltnew, ctlnew, ktnew, πicltnew, πccltnew, Beqltnew, ltnew_1, ltnew_2, Ωtlnew_1, Ωtlnew_2, πtlnew_1, πtlnew_2, ktlnew_1, ktlnew_2, ctlnew_1, ctlnew_2
        # else:
        #     return ltnew_1, ltnew_2, Ωtlnew_1, Ωtlnew_2, πtlnew_1, πtlnew_2, ktlnew_1, ktlnew_2, ctlnew_1, ctlnew_2
            
    
    def get_kt_nt(self, tt, ktnew, ntnew, aopt, nopt):
        nt = self.nt
        #Get aggregate capital and labor
        tp = tt-1
        i = 0
        while tp!=nt and i!=6:
            tp = tp+1
            i = i+1
            if tp>0: 
                ktnew[tp-1]=ktnew[tp-1]+1/6*aopt[i-1]
                if i<= 4: 
                    ntnew[tp-1] = ntnew[tp-1]+1/6*nopt[i-1]  
                    
        return ktnew, ntnew
    
    def create_spaces(self, Kbar1, Kbar0, Lbar1, Lbar0, Cbar1, Cbar0, Ω1, Ω0, τ_ic1 , τ_cc1, Beq1, Beq0, τpg1, rep1, pen_cc_2_1, pen_cc_2_0):
        nt, T, TR, ga, ϕ = self.nt, self.T, self.TR, self.ga, self.ϕ
        
        kst=np.zeros((nt,6))
        nst=np.zeros((nt,6))  
        
        ktold=np.zeros(nt)
        ktnew=np.zeros(nt)
        ntold=np.zeros(nt)
        ntnew=np.zeros(nt)
        ctnew=np.zeros(nt)
        Ωtnew=np.zeros(nt)
        ktold = self.seqa(Kbar0,(Kbar1-Kbar0)/(nt-1),nt)
        ltold = self.seqa(Lbar0,(Lbar1-Lbar0)/(nt-1),nt)
        ctold = self.seqa(Cbar0,(Cbar1-Cbar0)/(nt-1),nt)    
        Ωtold = self.seqa(Ω0,(Ω1-Ω0)/(nt-1),nt)    
        Beqtold = self.seqa(Beq0,(Beq1-Beq0)/(nt-1),nt) 
        pen_cc_t = self.seqa(pen_cc_2_0,(pen_cc_2_1-pen_cc_2_0)/(nt-1),nt) 
        xold=[ktold, ntold]    
        aopt = np.zeros(T+TR)
        τict=np.ones(nt)*τ_ic1
        τcct=np.ones(nt)*τ_cc1   
        τpgt=np.ones(nt)*τpg1  
        rept=np.ones(nt)*rep1
        
        #Productividad
        At = np.ones(nt)
        for i in range(1,nt):
            At[i] = At[i-1]*(1+ga)
        
        #Demografía
        m0, m1 = self.m0, self.m1
        Δm = (m1-m0)/(nt-1)
        mt = []
        mt.append([m0[0], m0[1], m0[2], m0[3], m0[4], m0[5]])
        for i in range(nt-1):
            res = mt[i] + Δm
            mt.append([res[0], res[1], res[2], res[3], res[4], res[5]])
        
        #Probabilidad de sobrevivencia
        ϕ0, ϕ1 = self.ϕ0, self.ϕ1
        Δϕ = (ϕ1-ϕ0)/(nt-1)
        ϕt = []
        ϕt.append([ϕ0[0], ϕ0[1], ϕ0[2], ϕ0[3], ϕ0[4], ϕ0[5]])
        for i in range(nt-1):
            res = ϕt[i] + Δϕ
            ϕt.append([res[0], res[1], res[2], res[3], res[4], res[5]])
                    
        
        return ktold, ltold, ctold, Ωtold, τict , τcct, Beqtold, τpgt, rept, At, mt, ϕt, pen_cc_t
    
    def get_AFP(self, τ_afp, w, r, N):
        
        afp1 = τ_afp*w*N[0]
        afp2 = τ_afp*w*N[1] + (1+r)*afp1
        afp3 = τ_afp*w*N[2] + (1+r)*afp2
        afp4 = τ_afp*w*N[3] + (1+r)*afp3
        return afp4
    
    
    def main(self):
        α, dep, τ0, τ1, tolk = self.α, self.dep, self.τ0, self.τ1, self.tolk
        nt, tolt, nqt, ϕ, T, TR = self.nt, self.tolt, self.nqt, self.ϕ, self.T, self.TR  
        τ_afp0, τ_afp1 = self.τ_afp0, self.τ_afp1   
        ρ, m, δ, τ_w, τ_r, τ_c, A, g, ga, ν = self.ρ, self.m, self.δ, self.τ_w, self.τ_r, self.τ_c, self.A, self.g, self.ga, self.ν
        me1, me2, e = self.me1, self.me2, self.e
        """ Type of model """
        μ = self.μ
        
        """ ---------------------- """
        '''Get Initial Steady State'''
        """ ---------------------- """
        
        #Update parameters
        self.A = 1.0
        self.m = self.m0
        self.ϕ = self.ϕ0
        m = self.m
        ϕ = self.ϕ
        
        #Solve steady state
        τ_ic0 , τ_cc0, τ_pg0 =  self.τ_ic0 , self.τ_cc0, self.τ_pg0
        self.pen_cc_2 = self.pen_cc_2_0
        pen_cc_2 = self.pen_cc_2
        ω1_0, ω2_0, l1_0, l2_0, C1_0, C2_0, Π_ic_1_0, Π_ic_2_0, Π_cc_1_0, Π_cc_2_0, w_0, r_0, pen_ic_1_0, pen_ic_2_0, pen_cc_1_0, pen_cc_2_0, pen_0 , pen_sp_2_0, Kbar_0, Lbar_0, Cbar_0, Πbar_0, Ωbar_0 = self.getss(τ_ic0, τ_cc0, τ_pg0)
        
        
        L1_0, L2_0, Ω1_0, Ω2_0, Π1_0, Π2_0, K1_0, K2_0, CC1_0, CC2_0 = self.getss(τ_ic0, τ_cc0, τ_pg0, prod=True)
        dic_ss_prod_0 = {'L1':L1_0, 'L2':L2_0, 'Ω1':Ω1_0, 'Ω2':Ω2_0, 
                         'Π1':Π1_0, 'Π2':Π2_0, 'K1':K1_0, 'K2':K2_0, 'CC1':CC1_0, 'CC2':CC2_0}

        """ ACA Revisar tasa de reemplazo, incluir eficiencia """ 
        inc_1 = (1-τ_w-μ*τ_ic0-μ*τ_cc0-(1-μ)*τ_pg0)*w_0*e[0][3] #*l1[3]
        inc_2 = (1-τ_w-μ*τ_ic0-μ*τ_cc0-(1-μ)*τ_pg0)*w_0*e[1][3] #*l2[3]    
                
        rep_tot_1_0 = (pen_ic_1_0 + pen_cc_1_0 + pen_0 )/inc_1
        rep_tot_2_0 = (pen_ic_2_0 + pen_cc_2_0 + pen_0 + pen_sp_2_0)/inc_2
        rep_tot_0 = me1*rep_tot_1_0 +  me2*rep_tot_2_0  #np.mean((rep_tot_1_0, rep_tot_2_0))
        
        PVU0 = np.sum(self.PVU(C1_0, l1_0))
        
        dic_ss0 = {'τ_ic': τ_ic0, 'τ_cc':τ_cc0, 'τ_pg':τ_pg0, 
                   'ω1':ω1_0, 'l1':l1_0, 'ω2':ω2_0, 'l2':l2_0, 'c1':C1_0, 'c2':C2_0,
                   'Kbar':Kbar_0, 'Lbar':Lbar_0, 'Cbar':Cbar_0,
                   'w':w_0, 'r':r_0, 
                   'pen_ic_1':pen_ic_1_0, 'pen_ic_2':pen_ic_2_0, 
                   'pen_cc_1':pen_cc_1_0, 'pen_cc_2':pen_cc_2_0, 
                   'pen_pg':pen_0, 'pen_sp':pen_sp_2_0, 
                   'Πbar':Πbar_0, 'Ωbar':Ωbar_0,
                   'Π_ic_1':Π_ic_1_0, 'Π_ic_2':Π_ic_2_0, 'Π_cc_1':Π_ic_1_0, 'Π_cc_2':Π_ic_2_0,
                   'rep_tot':rep_tot_0, 'rep_tot_1':rep_tot_1_0, 'rep_tot_2':rep_tot_2_0,'PVU':PVU0}
        

        #Government Bequest (ESTACIONARIO)
        Beq0 = self.Bequest(ω1_0, ω2_0, Lbar_0)
    
    
        """ ------------------ """
        '''Get New Steady State'''
        """ ------------------ """
        
        #Update parameters
        self.A = 1.*(1+ga)**(nt-1)
        self.m = self.m1
        # self.ϕ = self.ϕ1   
        m = self.m
        # ϕ = self.ϕ        

        #Solve steady state
        τ_ic1 , τ_cc1, τ_pg1 =  self.τ_ic1 , self.τ_cc1, self.τ_pg1 
        self.pen_cc_2 = self.pen_cc_2_1
        pen_cc_2 = self.pen_cc_2
        ω1_1, ω2_1, l1_1, l2_1, C1_1, C2_1, Π_ic_1_1, Π_ic_2_1, Π_cc_1_1, Π_cc_2_1, w_1, r_1, pen_ic_1_1, pen_ic_2_1, pen_cc_1_1, pen_cc_2_1, pen_1 , pen_sp_2_1, Kbar_1, Lbar_1, Cbar_1, Πbar_1, Ωbar_1 = self.getss(τ_ic1 , τ_cc1, τ_pg1)       
        
        L1_1, L2_1, Ω1_1, Ω2_1, Π1_1, Π2_1, K1_1, K2_1, CC1_1, CC2_1 = self.getss(τ_ic1, τ_cc1, τ_pg1, prod=True)
        dic_ss_prod_1 = {'L1':L1_1, 'L2':L2_1, 'Ω1':Ω1_1, 'Ω2':Ω2_1, 
                         'Π1':Π1_1, 'Π2':Π2_1, 'K1':K1_1, 'K2':K2_1, 'CC1':CC1_1, 'CC2':CC2_1}        
        

        """ ACA: incluir eficiencia """ 
        inc_1 = (1-τ_w-μ*τ_ic1-μ*τ_cc1-(1-μ)*τ_pg1)*w_1*e[0][3] #*l1[3]
        inc_2 = (1-τ_w-μ*τ_ic1-μ*τ_cc1-(1-μ)*τ_pg1)*w_1*e[1][3] #*l2[3]    
        
        rep_tot_1_1 = (pen_ic_1_1 + pen_cc_1_1 + pen_1)/inc_1
        rep_tot_2_1 = (pen_ic_2_1 + pen_cc_2_1 + pen_1 + pen_sp_2_1)/inc_2
        rep_tot_1 = me1*rep_tot_1_1 +  me2*rep_tot_2_1  # np.mean((rep_tot_1_1, rep_tot_2_1))
        rep1 = 2*(1-μ)*τ_pg1/(1-(1-μ)*τ_pg1) # REP sólo pay as you go (se ocupa en transición) 
        
        PVU1 = np.sum(self.PVU(C1_1, l1_1)) 
        
        dic_ss1 = {'τ_ic': τ_ic1, 'τ_cc':τ_cc1, 'τ_pg':τ_pg1, 
                   'ω1':ω1_1, 'l1':l1_1, 'ω2':ω2_1, 'l2':l2_1, 'c1':C1_1, 'c2':C2_1,
                   'Kbar':Kbar_1, 'Lbar':Lbar_1, 'Cbar':Cbar_1,
                   'w':w_1, 'r':r_1, 
                   'pen_ic_1':pen_ic_1_1, 'pen_ic_2':pen_ic_2_1, 
                   'pen_cc_1':pen_cc_1_1, 'pen_cc_2':pen_cc_2_1, 
                   'pen_pg':pen_1, 'pen_sp':pen_sp_2_1, 
                   'Π_ic_1':Π_ic_1_1, 'Π_ic_2':Π_ic_2_1, 'Π_cc_1':Π_cc_1_1, 'Π_cc_2':Π_cc_2_1,
                   'Πbar':Πbar_1, 'Ωbar':Ωbar_1,
                   'rep_tot':rep_tot_1, 'rep_tot_1':rep_tot_1_1, 'rep_tot_2':rep_tot_2_1,'PVU':PVU1}
        
                
        #Government Bequest (ESTACIONARIO)
        Beq1 = self.Bequest(ω1_1, ω2_1, Lbar_1)    

        """ ------------------ """
        '''     Transition'''
        """ ------------------ """
        
        ktold, ltold, Ctold, Ωtold, τict, τcct, Beqtold, τpgt, rept, At, mt, ϕt, pen_cc_2_t = self.create_spaces(Kbar_1, Kbar_0, Lbar_1, Lbar_0, Cbar_1, Cbar_0 , Ωbar_1, Ωbar_0 , τ_ic1 , τ_cc1, Beq1, Beq0, τ_pg1, rep1, pen_cc_2_1, pen_cc_2_0)
        # At = A #Update At from create_spaces
        self.At = At
        self.mt = mt
        self.ϕt = ϕt
        

        #Initialization 
#         update_t, update_new, update_old, ωlopt, states
        """ACA: VER SI HAY QUE AGREGAR PRODUCTIVIDAD 2 """
        ωlopt_new = np.concatenate((ω1_1, l1_1, ω2_1, l2_1))
    
        """ AGREGAR ESTADOS PRODUCTIVIDAD 2 """
        states_old = [C1_0, l1_0, ω1_0, C2_0, l2_0, ω2_0]
#         coptold_1, loptold_1, ωoptold_1 = states[0], states[1], states[2] 
        #Update prices, pensions, tau in stedy state
        update_old = [τ_ic0, τ_cc0, w_0, r_0, pen_0, τ_pg0, pen_sp_2_0, pen_cc_2_0]
        update_new = [τ_ic1, τ_cc1, w_1, r_1, pen_1, τ_pg1, pen_sp_2_1, pen_cc_2_1]

        #Capital convergence over transition
        q=0
        kritt=1+tolt    
        while q!=nqt and (q<=1 or kritt>=tolk):
        # for q in range(1):
            q = q+1

            #Transition prices given aggregate capital and labor (dim=20)
            # wt = (1-α) * ktold**α * ltold**(-α) # NO ESTACIONARIO
            # rt = α * ktold**(α-1)*ltold**(1-α) - δ #NO ESTACIONARIO
            wt = (1-α) * ktold**α  #ESTACIONARIO
            rt = α * ktold**(α-1) - δ #ESTACIONARIO
            
            #Pay as you go pension
            pent = rept * (1-(1-μ)*τpgt) * wt * ltold*3/2     
            
            #Goverment 
            Tax = τ_w*wt*ltold/(At*ltold) + τ_r*rt*Ωtold/ltold + τ_c*Ctold #C ya viene estacionario (por eso no es /L) 

            #Pensión solidaria
            tr = ν*(Tax + Beqtold - g*ktold**α)*ltold/(m[4]+m[5])   #TAX,  Beqtold y ktold ESTACIONARIOS  
            pen_sp_2t = tr/TR            

            """ AGREGAR pen_sp_2 """
            update_t = [τict, τcct, wt, rt, pent, τpgt, pen_sp_2t, pen_cc_2_t]  
            
            #Get aggregate capital and labor (dim=20)
            """ AGREGAR Beq """
            Ωtnew, ltnew, Ctnew, ktnew, πictnew, πcctnew, Beqtnew, ltnew_1, ltnew_2, Ωtlnew_1, Ωtlnew_2, πtlnew_1, πtlnew_2, ktlnew_1, ktlnew_2, ctlnew_1, ctlnew_2 = self.transition(update_t, update_new, update_old, ωlopt_new, states_old)
            # print("ktnew:", ktnew)

            #Convergence criteria
            kritt   = np.mean(abs(ktnew-ktold)) #+ np.mean(abs(ltnew-ltold)) + np.mean(abs(Ctnew-Ctold))#
            ktold   = ρ*ktold+(1-ρ)*ktnew
            ltold   = ρ*ltold+(1-ρ)*ltnew    
            Ctold   = ρ*Ctold+(1-ρ)*Ctnew   
            Ωtold   = ρ*Ωtold+(1-ρ)*Ωtnew
            Beqtold = ρ*Beqtold+(1-ρ)*Beqtnew
            Πt= πictnew + πcctnew
             #update de c y omega (para la actualización e los precios y pension solidaria)
        dic_trans = {'kt':ktnew, 'lt':ltnew, 'Ct':Ctnew, 'Ωt':Ωtnew, 'Πt':Πt, 'πic':πictnew, 'πcc':πcctnew}    
        dic_trans_prod = {'ltnew_1':ltnew_1, 'ltnew_2':ltnew_2, 'Ωtnew_1':Ωtlnew_1, 'Ωtnew_2':Ωtlnew_2, 'πtnew_1':πtlnew_1, 'πtnew_2':πtlnew_2, 'ktnew_1':ktlnew_1, 'ktnew_2':ktlnew_2, 'ctnew_1':ctlnew_1, 'ctnew_2':ctlnew_2}
        
        return dic_ss0, dic_ss1, dic_trans, dic_ss_prod_0, dic_ss_prod_1, dic_trans_prod
        
    
    
    
    def fun_get_steady_state(self, τ_ic, τ_cc, τ_pg):
    
        #Solve steady state
        # ω1, ω2, l1, l2, C1, C2, Π_ic_1, Π_ic_2, Π_cc, w, r, pen_ic_1, pen_ic_2, pen_cc_1, pen_cc_2, pen , pen_sp_2, Kbar, Lbar, Cbar = self.getss(τ_ic, τ_cc, τ_pg)
        
        ω1, ω2, l1, l2, C1, C2, Π_ic_1, Π_ic_2, Π_cc_1, Π_cc_2, w, r, pen_ic_1, pen_ic_2, pen_cc_1, pen_cc_2, pen , pen_sp_2, Kbar, Lbar, Cbar, Πbar, Ωbar = self.getss(τ_ic, τ_cc, τ_pg)
        
        τ_w, μ, e = self.τ_w, self.μ, self.e
        # inc_1 = (1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*e[0][3] #*l1[3]
        # inc_2 = (1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*e[1][3] #*l2[3]
        inc_1 = (1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*e[0]*l1
        inc_1 = np.mean(inc_1)
        inc_2 = (1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*e[1]*l2
        inc_2 = np.mean(inc_2)        
        # inc_1 = w*e[0][3]
        # inc_2 = w*e[1][3] 
        
#         pen_aux = (pen_ic_1 + pen_cc_1 + pen)/(e[0][3]*l1[3])
#         print("ratio:", pen_aux/w)
        
        rep_tot_1 = (pen_ic_1 + pen_cc_1 + pen)/inc_1
        rep_tot_2 = (pen_ic_2 + pen_cc_2 + pen + pen_sp_2)/inc_2
        
        me1, me2 = self.me1, self.me2
        rep_tot = rep_tot_1*me1 + rep_tot_2*me2 
        # rep_tot = np.mean((rep_tot_1, rep_tot_2))
        
        PVU1 = self.PVU(C1, l1)
        PVU2 = self.PVU(C2, l2)
        
        PVU1_s = sum(self.PVU(C1, l1))
        PVU2_s = sum(self.PVU(C2, l2))  
                
        A, α = self.A, self.α
        Y = A*Lbar**(1-α)*(Kbar*Lbar)**(α)
        Y = (A*Kbar)**(α)
                
        dic_ss = {'τ_ic': τ_ic, 'τ_cc':τ_cc, 'τ_pg':τ_pg, 
                   'ω1':ω1, 'l1':l1, 'ω2':ω2, 'l2':l2, 'c1':C1, 'c2':C2,
                   'Kbar':Kbar, 'Lbar':Lbar, 'Cbar':Cbar, 'Πbar':Πbar, 'Ωbar':Ωbar, 'Y':Y,
                   'w':w, 'r':r, 
                   'pen_ic_1':pen_ic_1, 'pen_ic_2':pen_ic_2, 
                   'pen_cc_1':pen_cc_1, 'pen_cc_2':pen_cc_2, 
                   'pen_pg':pen, 'pen_sp':pen_sp_2, 
                   'Π_ic_1':Π_ic_1, 'Π_ic_2':Π_ic_2, 'Π_cc_1':Π_cc_1, 'Π_cc_2':Π_cc_2, 
                   'rep_tot':rep_tot, 'rep_tot_1':rep_tot_1, 'rep_tot_2':rep_tot_2,
                   'PVU1':PVU1, 'PVU2':PVU2, 'PVU1_s':PVU1_s, 'PVU2_s':PVU2_s, 
                   'inc_1':inc_1, 'inc_2':inc_2}
        
        
        Π_1, Π_2 = self.fun_pension_agency(dic_ss, agg=False)
        dic_ss['Π_1'] = Π_1
        dic_ss['Π_2'] = Π_2
        
        return dic_ss
            
    
    
    #12. PLOTS
    def labor_suply_ss(self, nold, nnew):
        T = np.arange(1, len(nold)+1)
        fig, ax = plt.subplots()
        ax.plot(T, nold, linestyle='--', label='Old steady state')
        ax.plot(T, nnew, label='New steady state')
        ax.set(xlabel='Generation', ylabel='Labor Supply',
               title='Age-Labor Supply Profile')
        
        ax.grid()
        plt.show
        plt.legend()
        
    def labor_suply_generation(self, nold, nnew, tt):
        T = np.arange(1, len(nold)+1)
        fig, ax = plt.subplots()
        ax.plot(T, nold, linestyle='--', label='Old steady state')
        ax.plot(T, nnew, label='Individual born in t='+str(tt))
        ax.set(xlabel='Age', ylabel='Labor Supply',
               title='Age-Labor Supply Profile')
        
        ax.grid()
        plt.show
        plt.legend()

        
        
    def subplots_SS(self, dic_ss): 
        T, TR = self.T, self.TR
        
        
        var=["Age-labor supply", "Age-wealth", "Consumption", "PVU"]
        fig = make_subplots(rows=2, cols=2, subplot_titles=var)  
        
        colors = ['blue', 'red']
        t_lines = ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
        
        l1 = dic_ss['l1']
        l2 = dic_ss['l2']
        
        ω1 = dic_ss['ω1']
        ω2 = dic_ss['ω2']
        
        c1 = dic_ss['c1']
        c2 = dic_ss['c2']
        
        PVU1 = dic_ss['PVU1']
        PVU2 = dic_ss['PVU2']   
        
        fig.add_trace(go.Scatter(y=l1, mode='lines', showlegend=True, name='Alta', line=dict(color='red', dash=t_lines[1])), row=1, col=1)
        fig.add_trace(go.Scatter(y=l2, mode='lines', showlegend=True, name='Baja', line=dict(color='blue', dash=t_lines[2])), row=1, col=1)
        

        fig.add_trace(go.Scatter(y=ω1, mode='lines', showlegend=False, name='', line=dict(color='red', dash=t_lines[1])), row=1, col=2)
        fig.add_trace(go.Scatter(y=ω2, mode='lines', showlegend=False, name='', line=dict(color='blue', dash=t_lines[2])), row=1, col=2)
        
        fig.add_trace(go.Scatter(y=c1, mode='lines', showlegend=False, name='', line=dict(color='red', dash=t_lines[1])), row=2, col=1)
        fig.add_trace(go.Scatter(y=c2, mode='lines', showlegend=False, name='', line=dict(color='blue', dash=t_lines[2])), row=2, col=1)
        
        fig.add_trace(go.Scatter(y=PVU1, mode='lines', showlegend=False, name='', line=dict(color='red', dash=t_lines[1])), row=2, col=2)
        fig.add_trace(go.Scatter(y=PVU2, mode='lines', showlegend=False, name='', line=dict(color='blue', dash=t_lines[2])), row=2, col=2)        
        
        
        fig.update_layout(height=600, width=800)
        fig.show()   
    
        
    def subplots_SS_CI_plus(self, dic_ss_ci, dic_ss, 
                            path=False, 
                            name=False,
                            Label=False,
                            agg = True): 
        T, TR = self.T, self.TR
        me1, me2, m = self.me1, self.me2, self.m
        template = "plotly_white"# "simple_white"#"simple_white" 
        # template = dict(layout=go.Layout(title_font=dict(family="Rockwell", size=24)))
        
        """ Individual capitalization """
        l1_ci = dic_ss_ci['l1']; l2_ci = dic_ss_ci['l2']
        ω1_ci = dic_ss_ci['ω1']; ω2_ci = dic_ss_ci['ω2']
        c1_ci = dic_ss_ci['c1']; c2_ci = dic_ss_ci['c2']
        PVU1_ci = dic_ss_ci['PVU1']; PVU2_ci = dic_ss_ci['PVU2']  
        Π_1_ci = dic_ss_ci['Π_1']; Π_2_ci = dic_ss_ci['Π_2']  
        
        """ Other """
        l1 = dic_ss['l1']; l2 = dic_ss['l2']
        ω1 = dic_ss['ω1']; ω2 = dic_ss['ω2']
        c1 = dic_ss['c1']; c2 = dic_ss['c2']
        PVU1 = dic_ss['PVU1']; PVU2 = dic_ss['PVU2']                 
        Π_1 = dic_ss['Π_1']; Π_2 = dic_ss['Π_2']  
        
        if agg == True:
            """ Individual capitalization """
            l_ci = me1*l1_ci + me2*l2_ci #m[0:4]*
            c_ci = me1*c1_ci + me2*c2_ci #*m
            P_ci = me1*np.array(PVU1_ci) + me2*np.array(PVU2_ci)

            Π_ci = Π_1_ci + Π_2_ci #self.fun_pension_agency(dic_ss_ci)
            k_ci = me1*m*np.concatenate([[0], ω1_ci]) + me2*m*np.concatenate([[0], ω2_ci])  + Π_ci
            

            """ Other """
            l = me1*l1 + me2*l2 #m[0:4]*
            c = me1*c1 + me2*c2 #m*
            P = me1*np.array(PVU1) + me2*np.array(PVU2)

            Π = Π_1 + Π_2 #self.fun_pension_agency(dic_ss)
            k = me1*m*np.concatenate([[0], ω1]) + me2*m*np.concatenate([[0], ω2])  + Π
            
        
        else:
            """ Individual capitalization """
            l_ci_1 = me1*l1_ci 
            l_ci_2 = me2*l2_ci
            c_ci_1 = me1*c1_ci#*m 
            c_ci_2 = me2*c2_ci#*m
            P_ci_1 = me1*np.array(PVU1_ci) 
            P_ci_2 = me2*np.array(PVU2_ci)
            
            # Π_ci_1, Π_ci_2 = self.fun_pension_agency(dic_ss_ci, agg=agg)
            Ω_1_ci = me1*m*np.concatenate([[0], ω1_ci])
            Ω_2_ci = me2*m*np.concatenate([[0], ω2_ci])
            
            k_ci_1 = Ω_1_ci + Π_1_ci        
            k_ci_2 = Ω_2_ci + Π_2_ci    
            
            """ Other """
            l_1 = me1*l1 
            l_2 = me2*l2
            c_1 = me1*c1#*m
            c_2 = me2*c2#*m
            P_1 = me1*np.array(PVU1)
            P_2 = me2*np.array(PVU2)
            
            Ω_1 = me1*m*np.concatenate([[0], ω1])
            Ω_2 = me2*m*np.concatenate([[0], ω2])
            
            # Π_1, Π_2 = self.fun_pension_agency(dic_ss, agg=agg)
            k_1 = Ω_1 + Π_1
            k_2 = Ω_2 + Π_2
                        
           
        #######################################################################################
               
        colors = ['blue', 'red', '#00CC96']
        t_lines = ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']

        if agg == True:
            var=["(A) Labor supply", "(B) Capital", "(C) Consumption", "(D) Welfare"]
            fig = make_subplots(rows=2, cols=2, subplot_titles=var)  
            x = [1, 2, 3, 4, 5, 6]
 
            
            fig.add_trace(go.Scatter(x=x[0:4], y=l, mode='lines', showlegend=True, name=Label, line=dict(color='red', dash=t_lines[1])), row=1, col=1)
            fig.add_trace(go.Scatter(x=x[0:4], y=l_ci, mode='lines', showlegend=True, name='Individual capitalization', line=dict(color='blue', dash=t_lines[2])), row=1, col=1)

            fig.add_trace(go.Scatter(x=x, y=k, mode='lines', showlegend=False, name='', line=dict(color='red', dash=t_lines[1])), row=1, col=2)
            fig.add_trace(go.Scatter(x=x, y=k_ci, mode='lines', showlegend=False, name='', line=dict(color='blue', dash=t_lines[2])), row=1, col=2)

            fig.add_trace(go.Scatter(x=x, y=c, mode='lines', showlegend=False, name='', line=dict(color='red', dash=t_lines[1])), row=2, col=1)
            fig.add_trace(go.Scatter(x=x, y=c_ci, mode='lines', showlegend=False, name='', line=dict(color='blue', dash=t_lines[2])), row=2, col=1)

            fig.add_trace(go.Scatter(x=x, y=P/P_ci, mode='lines', showlegend=False, name='', line=dict(color='red', dash=t_lines[1])), row=2, col=2)
            # fig.add_trace(go.Scatter(x=x, y=P/P_ci, mode='lines', showlegend=False, name='', line=dict(color='blue', dash=t_lines[2])), row=2, col=2)  
            
            # edit axis labels
            fig['layout']['xaxis']['title']='Age'
            fig['layout']['xaxis2']['title']='Age'
            fig['layout']['xaxis3']['title']='Age'
            fig['layout']['xaxis4']['title']='Age'        
            fig['layout']['yaxis']['title']='Labor'
            fig['layout']['yaxis2']['title']='Capital'        
            fig['layout']['yaxis3']['title']='Consumption'
            fig['layout']['yaxis4']['title']='PVU'

            fig['layout']['xaxis2']['dtick']=1
            fig['layout']['xaxis3']['dtick']=1
            fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
            fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
            fig.update_layout(template=template)
            fig.update_layout(height=600, width=900)
            fig.update_layout(legend=dict(
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ))     
            
        elif agg == "welfare":
            var = []
            if Label==False:
                var=[""]
            else:
                var=[Label]
                
            fig = make_subplots(rows=1, cols=1, subplot_titles=var)  
            x = [1, 2, 3, 4, 5, 6]
            
            # print("IC", (m@P_ci_1)*me1, m@P_ci_2*me2, (m@P_ci_1)*me1 + m@P_ci_2*me2)
            # print("Otro", (m@P_1)*me1, m@P_2*me2, (m@P_1)*me1 + m@P_2*me2)
            m_r = m/np.sum(m)
            print("W:", me1*(m_r@(P_1/P_ci_1)) + me2*(m_r@(P_2/P_ci_2)))

            # fig.add_trace(go.Scatter(x=x, y=P_1/P_ci_1, mode='lines', showlegend=False, name='', line=dict(color='blue', dash=t_lines[1])), row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=P_1/P_ci_1, mode='lines', showlegend=True, name='High productivity', line=dict(color='red', dash=t_lines[2])), row=1, col=1)              
            
            fig.add_trace(go.Scatter(x=x, y=P_2/P_ci_2, mode='lines', showlegend=True, name='Low productivity', line=dict(color='gray', dash=t_lines[0])), row=1, col=1)            

            
            # edit axis labels
            fig['layout']['xaxis']['title']='Age'     
            fig['layout']['yaxis']['title']="Welfare index"

            fig['layout']['xaxis']['dtick']=1
            fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
            fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
            fig.update_layout(template=template)
            fig.update_layout(height=500, width=500)
            fig.update_layout(legend=dict(
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ))                 

        else:
            
            var=["(A) Consumption", "(B) Capital", "(C) Pension funds", "(D) Savings"]
            fig = make_subplots(rows=2, cols=2, subplot_titles=var)  
            x = [1, 2, 3, 4, 5, 6]  
            colors=['red', 'blue', '#990099', '#1CBE4F']
            colors=['red', 'gray', 'purple', 'Orange']
#             fig.add_trace(go.Scatter(x=x[0:4], y=l_1, mode='lines', showlegend=False, name=Label, line=dict(color='red', dash=t_lines[0])), row=3, col=1)
#             fig.add_trace(go.Scatter(x=x[0:4], y=l_ci_1, mode='lines', showlegend=False, name='Individual capitalization', line=dict(color='blue', dash=t_lines[1])), row=3, col=1)
            
#             fig.add_trace(go.Scatter(x=x[0:4], y=l_2, mode='lines', showlegend=False, name=Label, line=dict(color='#990099', dash=t_lines[0])), row=3, col=1)
#             fig.add_trace(go.Scatter(x=x[0:4], y=l_ci_2, mode='lines', showlegend=False, name='Individual capitalization', line=dict(color='#1CBE4F', dash=t_lines[1])), row=3, col=1)           

            fig.add_trace(go.Scatter(x=x, y=c_1, mode='lines', showlegend=False, name='', line=dict(color=colors[0], dash=t_lines[0])), row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=c_ci_1, mode='lines', showlegend=False, name='', line=dict(color=colors[1], dash=t_lines[1])), row=1, col=1)

            fig.add_trace(go.Scatter(x=x, y=c_2, mode='lines', showlegend=False, name='', line=dict(color=colors[2], dash=t_lines[0])), row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=c_ci_2, mode='lines', showlegend=False, name='', line=dict(color=colors[3], dash=t_lines[2])), row=1, col=1)   


            fig.add_trace(go.Scatter(x=x, y=k_1, mode='lines', showlegend=True, name='High productivity (' +Label + ')', line=dict(color=colors[0], dash=t_lines[0])), row=1, col=2)
            fig.add_trace(go.Scatter(x=x, y=k_ci_1, mode='lines', showlegend=True, name='High productivity (FDC)', line=dict(color=colors[1], dash=t_lines[1])), row=1, col=2)
            
            fig.add_trace(go.Scatter(x=x, y=k_2, mode='lines', showlegend=True, name='Low productivity ('+Label + ')', line=dict(color=colors[2], dash=t_lines[0])), row=1, col=2)
            fig.add_trace(go.Scatter(x=x, y=k_ci_2, mode='lines', showlegend=True, name='Low productivity (FDC)', line=dict(color=colors[3], dash=t_lines[2])), row=1, col=2)            

         
            
            fig.add_trace(go.Scatter(x=x, y=Π_1, mode='lines', showlegend=False, name='', line=dict(color=colors[0], dash=t_lines[0])), row=2, col=1)
            fig.add_trace(go.Scatter(x=x, y=Π_1_ci, mode='lines', showlegend=False, name='', line=dict(color=colors[1], dash=t_lines[1])), row=2, col=1)   
            
            fig.add_trace(go.Scatter(x=x, y=Π_2, mode='lines', showlegend=False, name='', line=dict(color=colors[2], dash=t_lines[0])), row=2, col=1)
            fig.add_trace(go.Scatter(x=x, y=Π_2_ci, mode='lines', showlegend=False, name='', line=dict(color=colors[3], dash=t_lines[2])), row=2, col=1)            
            
            
            fig.add_trace(go.Scatter(x=x, y=Ω_1, mode='lines', showlegend=False, name='', line=dict(color=colors[0], dash=t_lines[0])), row=2, col=2)
            fig.add_trace(go.Scatter(x=x, y=Ω_1_ci, mode='lines', showlegend=False, name='', line=dict(color=colors[1], dash=t_lines[1])), row=2, col=2)   
            
            fig.add_trace(go.Scatter(x=x, y=Ω_2, mode='lines', showlegend=False, name='', line=dict(color=colors[2], dash=t_lines[0])), row=2, col=2)
            fig.add_trace(go.Scatter(x=x, y=Ω_2_ci, mode='lines', showlegend=False, name='', line=dict(color=colors[3], dash=t_lines[2])), row=2, col=2)     
            
            

            # edit axis labels
            fig['layout']['xaxis']['title']='Age'
            fig['layout']['xaxis2']['title']='Age'
            fig['layout']['xaxis3']['title']='Age'
            fig['layout']['xaxis4']['title']='Age'        
            fig['layout']['yaxis']['title']='Consumption'
            fig['layout']['yaxis2']['title']='Capital'        
            fig['layout']['yaxis3']['title']='Pension funds'
            fig['layout']['yaxis4']['title']='Savings'

            fig['layout']['xaxis']['dtick']=1
            fig['layout']['xaxis2']['dtick']=1
            # fig['layout']['xaxis3']['dtick']=1
            
            fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
            fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
            fig.update_layout(template=template)
            fig.update_layout(height=700, width=900)
            fig.update_layout(legend=dict(
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            ))         
        
        if path == False:
            fig.show()
        else:
            fig.show()
            fig.write_image(path + name)
            
            
    def subplots_transition(self, dic_ss0, dic_ss1, dic_trans, 
                            dic_ss_prod_0, dic_ss_prod_1, dic_trans_prod,
                            path=False, 
                            name=False,
                            Label=False): 
        T, TR, nt = self.T, self.TR, self.nt
        
        template = "plotly_white"

        var=["(A) Capital", "(B) Pension funds", "(C) Savings", "(D) Labor", "(E) Consumption"]
        fig = make_subplots(rows=3, cols=2, subplot_titles=var)  
        
        colors = ['blue', 'red', 'gray']
        t_lines = ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
        
        Kbar_0, Kbar_1, ktnew = dic_ss0['Kbar'], dic_ss1['Kbar'], dic_trans['kt']
        Lbar_0, Lbar_1, ltnew = dic_ss0['Lbar'], dic_ss1['Lbar'], dic_trans['lt']
        Cbar_0, Cbar_1, Ctnew = dic_ss0['Cbar'], dic_ss1['Cbar'], dic_trans['Ct']
        Πbar_0, Πbar_1, Πtnew = dic_ss0['Πbar'], dic_ss1['Πbar'], dic_trans['Πt']
        Ωbar_0, Ωbar_1, Ωtnew = dic_ss0['Ωbar'], dic_ss1['Ωbar'], dic_trans['Ωt']
        
        #Variables por tipo de productividad
        Kbar_1_0, Kbar_2_0 = dic_ss_prod_0['K1'], dic_ss_prod_0['K2']
        Lbar_1_0, Lbar_2_0 = dic_ss_prod_0['L1'], dic_ss_prod_0['L2']
        Cbar_1_0, Cbar_2_0 = dic_ss_prod_0['CC1'], dic_ss_prod_0['CC2']
        Πbar_1_0, Πbar_2_0 = dic_ss_prod_0['Π1'], dic_ss_prod_0['Π2']
        Ωbar_1_0, Ωbar_2_0 = dic_ss_prod_0['Ω1'], dic_ss_prod_0['Ω2']
                
        Kbar_1_1, Kbar_2_1 = dic_ss_prod_1['K1'], dic_ss_prod_1['K2']
        Lbar_1_1, Lbar_2_1 = dic_ss_prod_1['L1'], dic_ss_prod_1['L2']
        Cbar_1_1, Cbar_2_1 = dic_ss_prod_1['CC1'], dic_ss_prod_1['CC2']
        Πbar_1_1, Πbar_2_1 = dic_ss_prod_1['Π1'], dic_ss_prod_1['Π2']
        Ωbar_1_1, Ωbar_2_1 = dic_ss_prod_1['Ω1'], dic_ss_prod_1['Ω2']
                    
        ktnew_1, ktnew_2 = dic_trans_prod['ktnew_1'], dic_trans_prod['ktnew_2']
        ltnew_1, ltnew_2 = dic_trans_prod['ltnew_1'], dic_trans_prod['ltnew_2']
        Ctnew_1, Ctnew_2 = dic_trans_prod['ctnew_1'], dic_trans_prod['ctnew_2']
        Πtnew_1, Πtnew_2 = dic_trans_prod['πtnew_1'], dic_trans_prod['πtnew_2']
        Ωtnew_1, Ωtnew_2 = dic_trans_prod['Ωtnew_1'], dic_trans_prod['Ωtnew_2']
        
        K = np.concatenate([[Kbar_0], ktnew, [Kbar_1]])
        Π = np.concatenate([[Πbar_0], Πtnew, [Πbar_1]])
        Ω = np.concatenate([[Ωbar_0], Ωtnew, [Ωbar_1]])
        L = np.concatenate([[Lbar_0], ltnew, [Lbar_1]])
        C = np.concatenate([[Cbar_0], Ctnew, [Cbar_1]])
        
        K1 = np.concatenate([[Kbar_1_0], ktnew_1, [Kbar_1_1]])
        Π1 = np.concatenate([[Πbar_1_0], Πtnew_1, [Πbar_1_1]])
        Ω1 = np.concatenate([[Ωbar_1_0], Ωtnew_1, [Ωbar_1_1]])
        L1 = np.concatenate([[Lbar_1_0], ltnew_1, [Lbar_1_1]])
        C1 = np.concatenate([[Cbar_1_0], Ctnew_1, [Cbar_1_1]])
        
        K2 = np.concatenate([[Kbar_2_0], ktnew_2, [Kbar_2_1]])
        Π2 = np.concatenate([[Πbar_2_0], Πtnew_2, [Πbar_2_1]])
        Ω2 = np.concatenate([[Ωbar_2_0], Ωtnew_2, [Ωbar_2_1]])
        L2 = np.concatenate([[Lbar_2_0], ltnew_2, [Lbar_2_1]])
        C2 = np.concatenate([[Cbar_2_0], Ctnew_2, [Cbar_2_1]])
   
        # L = self.me1*L1 + self.me2*L2
        
        nt = self.nt
        T = np.arange(2020, 2020 + nt+8)
        nn = len(T)
        tt = 0
        #Plot Individual capitalization
        # fig.add_trace(go.Scatter(x=T[0:nn-tt], y=K[0:nn-tt], mode='lines', showlegend=True, name='Aggregate', line=dict(color=colors[0], dash=t_lines[0])), row=1, col=1)
        fig.add_trace(go.Scatter(x=T[0:nn-tt], y=K1[0:nn-tt], mode='lines', showlegend=True, name='High productivity', line=dict(color=colors[1], dash=t_lines[2])), row=1, col=1)
        fig.add_trace(go.Scatter(x=T[0:nn-tt], y=K2[0:nn-tt], mode='lines', showlegend=True, name='Low productivity', line=dict(color=colors[2], dash=t_lines[2])), row=1, col=1)        
        
        # fig.add_trace(go.Scatter(x=T[0:nn-tt], y=Π[0:nn-tt], mode='lines', showlegend=False, name='', line=dict(color=colors[0], dash=t_lines[0])), row=1, col=2)
        fig.add_trace(go.Scatter(x=T[0:nn-tt], y=Π1[0:nn-tt], mode='lines', showlegend=False, name='', line=dict(color=colors[1], dash=t_lines[2])), row=1, col=2)
        fig.add_trace(go.Scatter(x=T[0:nn-tt], y=Π2[0:nn-tt], mode='lines', showlegend=False, name='', line=dict(color=colors[2], dash=t_lines[2])), row=1, col=2)        

        # fig.add_trace(go.Scatter(x=T[0:nn-tt], y=Ω[0:nn-tt], mode='lines', showlegend=False, name='', line=dict(color=colors[0], dash=t_lines[0])), row=2, col=1)
        fig.add_trace(go.Scatter(x=T[0:nn-tt], y=Ω1[0:nn-tt], mode='lines', showlegend=False, name='', line=dict(color=colors[1], dash=t_lines[2])), row=2, col=1)
        fig.add_trace(go.Scatter(x=T[0:nn-tt], y=Ω2[0:nn-tt], mode='lines', showlegend=False, name='', line=dict(color=colors[2], dash=t_lines[2])), row=2, col=1)        
        
        # fig.add_trace(go.Scatter(x=T[0:nn-tt], y=L[0:nn-tt], mode='lines', showlegend=False, name='', line=dict(color=colors[0], dash=t_lines[0])), row=2, col=2)
        fig.add_trace(go.Scatter(x=T[0:nn-tt], y=L1[0:nn-tt], mode='lines', showlegend=False, name='', line=dict(color=colors[1], dash=t_lines[2])), row=2, col=2)
        fig.add_trace(go.Scatter(x=T[0:nn-tt], y=L2[0:nn-tt], mode='lines', showlegend=False, name='', line=dict(color=colors[2], dash=t_lines[2])), row=2, col=2)        
        
        # fig.add_trace(go.Scatter(x=T[0:nn-tt], y=C[0:nn-tt], mode='lines', showlegend=False, name='', line=dict(color=colors[0], dash=t_lines[0])), row=3, col=1)
        fig.add_trace(go.Scatter(x=T[0:nn-tt], y=C1[0:nn-tt], mode='lines', showlegend=False, name='', line=dict(color=colors[1], dash=t_lines[2])), row=3, col=1)
        fig.add_trace(go.Scatter(x=T[0:nn-tt], y=C2[0:nn-tt], mode='lines', showlegend=False, name='', line=dict(color=colors[2], dash=t_lines[2])), row=3, col=1)        
        
        # fig.add_trace(go.Scatter(x=T, y=c_ci, mode='lines', showlegend=False, name='', line=dict(color='blue', dash=t_lines[1])), row=2, col=1)
        
        # edit axis labels
        fig['layout']['xaxis']['title']='time'
        fig['layout']['xaxis2']['title']='time'
        fig['layout']['xaxis3']['title']='time'
        fig['layout']['xaxis4']['title']='time'    
        fig['layout']['xaxis5']['title']='time'   
        # fig['layout']['xaxis6']['title']='time'   
        fig['layout']['yaxis']['title']='Capital'
        fig['layout']['yaxis2']['title']='Pension funds'        
        fig['layout']['yaxis3']['title']='Savings'
        fig['layout']['yaxis4']['title']='Labor'
        fig['layout']['yaxis5']['title']='Consumption'
        
        # fig['layout']['xaxis2']['dtick']=1
        # fig['layout']['xaxis3']['dtick']=1
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_layout(height=900, width=900)
        fig.update_layout(template=template)
        fig.update_layout(legend=dict(
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        ))         
        
        if path == False:
            fig.show()
        else:
            fig.show()
            fig.write_image(path + name)            
            
            
        

    def capital_stock_ss(self, aold, anew):
        T = np.arange(1, len(aold)+1)
        fig, ax = plt.subplots()
        ax.plot(T, aold, linestyle='--', label='Old steady state')
        ax.plot(T, anew, label='New steady state')
        ax.set(xlabel='Generation', ylabel='Individual capital stock',
               title='Age-Capital Profile')
        
        ax.grid()
        plt.show
        plt.legend()
        
        
        
        
    def consumption_ss(self, cold, cnew):
        T = np.arange(1, len(cold)+1)
        fig, ax = plt.subplots()
        ax.plot(T, cold, linestyle='--', label='Old steady state')
        ax.plot(T, cnew, label='New steady state')
        ax.set(xlabel='Generation', ylabel='Individual consumption',
               title='Age-Consumption Profile')
        
        ax.grid()
        plt.show
        plt.legend()    
        
    def capital_stock_generation(self, aold, anew, tt):
        T = np.arange(1, len(aold)+1)
        fig, ax = plt.subplots()
        ax.plot(T, aold, linestyle='--', label='Old steady state')
        ax.plot(T, anew, label='Individual born in t='+str(tt))
        ax.set(xlabel='Age', ylabel='Individual capital stock',
               title='Age-Capital Profile')
        
        ax.grid()
        plt.show
        plt.legend()

    def transition_plot(self, kbarold, ktnew, kbarnew, name=""): 
        nt = self.nt
        T = np.arange(-3, nt+1+4)
        kt_old = np.ones(4) * kbarold
        kt_new = np.ones(4) * kbarnew
        kt = kt_old.tolist() + ktnew.tolist() + kt_new.tolist()
        
        fig, ax = plt.subplots()
        ax.plot(T, kt)

        ax.set(xlabel='Transition period', ylabel=name,
                   title=name)
     

        ax.grid()
        plt.show  
        #print("kbarold:", kbarold)
        #print("ktnew:", ktnew)
        #print(kt)
        
    def hev(self,copt,noptt,c0,n0):
        β =self.β
        u =self.u
        nt=self.nt
        s=self.s
        tt=nt+1
        Ut=np.zeros(tt+4)       
        U0=np.zeros(1) 
        U0=u(c0[0],1-n0[0])+β*u(c0[1],1-n0[1])+(β**2)*u(c0[2],1-n0[2])+(β**3)*u(c0[3],1-n0[3])+(β**4)*u(c0[4],1)+(β**5)*u(c0[5],1)


        
        for it in range(tt+4):
            Ut[it]=u(copt[it,0],1-noptt[it,0])+β*u(copt[it,1],1-noptt[it,1])+(β**2)*u(copt[it,2],1-noptt[it,2])+(β**3)*u(copt[it,3],1-noptt[it,3])+(β**4)*u(copt[it,4],1)+(β**5)*u(copt[it,5],1)
            
            
        RU=(Ut-U0)/np.abs(U0)
        RU1=(Ut/U0)**(1/(1-s))-1

        return Ut, U0, RU, RU1

    
    def inefficiency(self,Kbar0, Nbar0, ktnew, ntnew):
        α=self.α
        nt=self.nt
        tt=nt
        Yt=np.zeros(tt)
        
        Ybar0=Kbar0**(α)*Nbar0**(1-α)
        
        for it in range(tt):
            Yt[it]=ktnew[it]**(α)*ntnew[it]**(1-α)
            
        inefft=(Yt-Ybar0)/np.abs(Ybar0)

        return Ybar0, Yt, inefft
    
    
    def PVU(self, C, N):
        T, TR, β = self.T, self.TR, self.β
        PVU = []
        for i in range(T+TR):
            if i>=T:
                PVU.append(β**(i)*self.u(C[i], 1))
            else:
                PVU.append(β**(i)*self.u(C[i], 1-N[i]))
        
        return PVU
            
        
    def table(self, dic_ss0, dic_ss1):
        T, TR = self.T, self.TR
        names = ["Casos:", "τ_pg", "τ_ic", "rep", "rep_tot", "K", "N", "C", "PVU"]
        
        casos, τ_pg, τ_ic, rep, rep_tot, K, N, C, PVU = [], [], [], [], [], [], [], [], []

        casos = ["SS0", "SS1"]
        τ_pg = [round(dic_ss0['τ_pg'], 2), round(dic_ss1['τ_pg'], 2)]
        τ_ic = [round(dic_ss0['τ_ic'], 2), round(dic_ss1['τ_ic'], 2)]
        rep = [round(dic_ss0['rep'], 2), round(dic_ss1['rep'], 2)]
        rep_tot = [round(dic_ss0['rep_tot'], 2), round(dic_ss1['rep_tot'], 2)]
        K = [round(dic_ss0['Kbar'], 3), round(dic_ss1['Kbar'], 3)]
        N = [round(dic_ss0['Lbar'], 3), round(dic_ss1['Lbar'], 3)]
        C = [round(dic_ss0['Cbar'], 3), round(dic_ss1['Cbar'], 3)]
        PVU = [round(dic_ss0['PVU'], 2), round(dic_ss1['PVU'], 2)]
        
        fig = go.Figure(data=[go.Table(header=dict(values=names),
                         cells=dict(values=[casos, τ_pg, τ_ic, rep, rep_tot, K, N, C, PVU]))
                             ])
        
        fig.update_layout(width=600, height=300)
        fig.show()        
        
        
    def table_ef(self, dic_ss):
        T, TR = self.T, self.TR
        μ, τ_w, e = self.μ, self.τ_w, self.e
        τ_ic, τ_cc, τ_pg = dic_ss['τ_ic'], dic_ss['τ_cc'], dic_ss['τ_pg']
        
        def f_round(x): return round(x, 2)

        w   = dic_ss['w']
        l_1 = dic_ss['l1']
        l_2 = dic_ss['l2']
        
        ing_1_1 = (1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l_1[0]*e[0][0]    
        ing_2_1 = (1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l_1[1]*e[0][1]  
        ing_3_1 = (1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l_1[2]*e[0][2]  
        ing_4_1 = (1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l_1[3]*e[0][3]
        
        ing_1_2 = (1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l_2[0]*e[1][0]    
        ing_2_2 = (1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l_2[1]*e[1][1]  
        ing_3_2 = (1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l_2[2]*e[1][2]  
        ing_4_2 = (1-τ_w-μ*τ_ic-μ*τ_cc-(1-μ)*τ_pg)*w*l_2[3]*e[1][3]
        
        rep_tot_1 = dic_ss['rep_tot_1']
        rep_tot_2 = dic_ss['rep_tot_2']
        ω1 = dic_ss['ω1']
        ω2 = dic_ss['ω2']
        l1 = dic_ss['l1']
        l2 = dic_ss['l2']
        c1 = dic_ss['c1']
        c2 = dic_ss['c2']
        
        PVU1 = dic_ss['PVU1_s']
        PVU2 = dic_ss['PVU2_s']
        
        names = ["Efficiency", "Ing_1", "Ing_2", "Ing_3","Ing_4", "rep", "Ω", "L", "C", "PVU"]
        Productividad = ["High (1)", "Low (2)", "(1)/(2)"]
        rep_tot = [f_round(rep_tot_1), f_round(rep_tot_2), f_round(rep_tot_1/rep_tot_2)]
        ing_1   = [f_round(ing_1_1), f_round(ing_1_2), f_round(ing_1_1/ing_1_2)]
        ing_2   = [f_round(ing_2_1), f_round(ing_2_2), f_round(ing_2_1/ing_2_2)]
        ing_3   = [f_round(ing_3_1), f_round(ing_3_2), f_round(ing_3_1/ing_3_2)]
        ing_4   = [f_round(ing_4_1), f_round(ing_4_2), f_round(ing_4_1/ing_4_2)]
        
        K = [f_round(sum(ω1)), f_round(sum(ω2)), f_round(sum(ω1)/sum(ω2))]
        N = [f_round(sum(l1)), f_round(sum(l2)), f_round(sum(l1)/sum(l2))]
        C = [f_round(sum(c1)), f_round(sum(c2)), f_round(sum(c1)/sum(c2))]
        PVU = [f_round(PVU1), f_round(PVU2), f_round(PVU1/PVU2)]
        
        fig = go.Figure(data=[go.Table(header=dict(values=names),
                         cells=dict(values=[Productividad, ing_1, ing_2, ing_3, ing_4, rep_tot, K, N, C, PVU]))
                             ])
        
        fig.update_layout(width=900, height=300)
        fig.show()          

    def fun_get_agg(self, dic_ss, ratios=False): 
        Ωbar = dic_ss['Ωbar']
        Πbar = dic_ss['Πbar']
        Kbar = dic_ss['Kbar'] 
        Lbar = dic_ss['Lbar']
        Cbar = dic_ss['Cbar']
        Y = dic_ss['Y'] 
        
        w = dic_ss['w']
        r = dic_ss['r']   
  
        inc_1 = dic_ss['inc_1']
        inc_2 = dic_ss['inc_2']
        
        pen1 = dic_ss['pen_ic_1'] + dic_ss['pen_cc_1'] +  dic_ss['pen_pg'] 
        pen2 = dic_ss['pen_ic_2'] + dic_ss['pen_cc_2'] +  dic_ss['pen_pg'] +  dic_ss['pen_sp']
        pen_tot = pen1*self.me1 + pen2*self.me2
        
        rep_1 = dic_ss['rep_tot_1'] 
        rep_2 = dic_ss['rep_tot_2']
        Rep = dic_ss['rep_tot']
        Ratio_pen = pen1 / pen2
        #Ratios
        ΩY = Ωbar/Y
        KY = Kbar/Y
        KL = Kbar/Lbar
        rKLw = r*KY/(Lbar*w)
        rKY = r*Kbar/Y
        wLY = w*Lbar/Y
        ΠΩ_K = (Πbar+Ωbar)/Kbar
        Π_Y = Πbar/Y
        rKY_wLY = rKY+ wLY

        if ratios == False:
            return Ωbar, Kbar, Lbar, Cbar, Rep, Ratio_pen, w, r, pen1, pen2, pen_tot, Πbar, inc_1, inc_2, rep_1, rep_2
        else:
            return ΩY, KY, KL, rKLw, rKY, wLY, ΠΩ_K, rKY_wLY, Π_Y
        
        
    def fun_pension_agency(self, dic_ss, agg=True):
        me1, me2, m = self.me1, self.me2, self.m
        """ Agencia de pensiones """
        #Individual capitalization 
        π_ic_4_1 = np.array(dic_ss['Π_ic_1'][3])
        π_ic_4_2 = np.array(dic_ss['Π_ic_2'][3])
        
        ing_ic1 = π_ic_4_1*me1*m[4]
        ing_ic2 = π_ic_4_2*me2*m[4]        
        
        pen_ic_1 = ing_ic1/(me1*(m[4]+m[5]))
        pen_ic_2 = ing_ic2/(me2*(m[4]+m[5]))
        
        # Collective capitalization 
        π_cc_4_1 = np.array(dic_ss['Π_cc_1'][3])
        π_cc_4_2 = np.array(dic_ss['Π_cc_2'][3])        
        pen_cc_2 = self.pen_cc_2 #ahora es valor fijo #(π_cc_4_1 + π_cc_4_2)/(2*TR)
        ing_cc1 = π_cc_4_1*me1*m[4]
        ing_cc2 = π_cc_4_2*me2*m[4]
        
        cost_cc2_4 = pen_cc_2*me2*m[4]
        cost_cc2_5 = pen_cc_2*me2*m[5]
        
        FR_cc = (ing_cc1+ing_cc2) - (cost_cc2_4+cost_cc2_5)
        pen_cc_1 = FR_cc/(me1*(m[4]+m[5]))         
        
        """ Aggregate pension funds """
        #Individual capitalization
        cost_ic1_4 = pen_ic_1*me1*m[4]
        cost_ic1_5 = pen_ic_1*me1*m[5]
        
        cost_ic2_4 = pen_ic_2*me2*m[4]
        cost_ic2_5 = pen_ic_2*me2*m[5]
        
        Π_ic_1_5 = ing_ic1 - cost_ic1_4
        Π_ic_1_6 = Π_ic_1_5 - cost_ic1_5 
        
        Π_ic_2_5 = ing_ic2 - cost_ic2_4
        Π_ic_2_6 = Π_ic_2_5 - cost_ic2_5   
        
        π_ic_1_1, π_ic_2_1, π_ic_3_1, π_ic_4_1 = dic_ss['Π_ic_1'][0], dic_ss['Π_ic_1'][1], dic_ss['Π_ic_1'][2], dic_ss['Π_ic_1'][3]
        π_ic_1_2, π_ic_2_2, π_ic_3_2, π_ic_4_2 = dic_ss['Π_ic_2'][0], dic_ss['Π_ic_2'][1], dic_ss['Π_ic_2'][2], dic_ss['Π_ic_2'][3]
        
        
        Π_ic_1 = np.array([π_ic_1_1*m[0]*me1, π_ic_2_1*m[1]*me1, π_ic_3_1*m[2]*me1, π_ic_4_1*m[3]*me1, Π_ic_1_5, Π_ic_1_6])
        Π_ic_2 = np.array([π_ic_1_2*m[0]*me2, π_ic_2_2*m[1]*me2, π_ic_3_2*m[2]*me2, π_ic_4_2*m[3]*me2, Π_ic_2_5, Π_ic_2_6])     
        
        Π_ic = Π_ic_1 + Π_ic_2
        
        #Collective capitalization
        cost_cc1_4 = pen_cc_1*me1*m[4]
        cost_cc1_5 = pen_cc_1*me1*m[5]        
        Π_cc_1_5 = ing_cc1 - cost_cc1_4
        Π_cc_1_6 = Π_cc_1_5 - cost_cc1_5 
        
        Π_cc_2_5 = ing_cc2 - cost_cc2_4
        Π_cc_2_6 = Π_cc_2_5 - cost_cc2_5 
        
        π_cc_1_1, π_cc_2_1, π_cc_3_1, π_cc_4_1 = dic_ss['Π_cc_1'][0], dic_ss['Π_cc_1'][1], dic_ss['Π_cc_1'][2], dic_ss['Π_cc_1'][3]
        π_cc_1_2, π_cc_2_2, π_cc_3_2, π_cc_4_2 = dic_ss['Π_cc_2'][0], dic_ss['Π_cc_2'][1], dic_ss['Π_cc_2'][2], dic_ss['Π_cc_2'][3]
        
        Π_cc_1 = np.array([π_cc_1_1*m[0]*me1, π_cc_2_1*m[1]*me1, π_cc_3_1*m[2]*me1, π_cc_4_1*m[3]*me1, Π_cc_1_5, Π_cc_1_6])
        Π_cc_2 = np.array([π_cc_1_2*m[0]*me2, π_cc_2_2*m[1]*me2, π_cc_3_2*m[2]*me2, π_cc_4_2*m[3]*me2, Π_cc_2_5, Π_cc_2_6])     
        
        Π_cc = Π_cc_1 + Π_cc_2
    
        if agg == True:
            return Π_ic + Π_cc
        else: 
            return Π_ic_1 + Π_cc_1, Π_ic_2 + Π_cc_2
            