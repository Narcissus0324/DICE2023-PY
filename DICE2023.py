from math import exp,log
import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds, dual_annealing

class DICEModel:
    def __init__(self):
        # Define the time periods
        self.timesteps = range(1, 82)  # Assuming 81 time periods

        # Population and technology
        self.gama = 0.300
        self.pop1 = 7752.9
        self.popadj = 0.145
        self.popasym = 10825.0
        self.dk = 0.1
        self.q1 = 135.7
        self.AL1 = 5.84
        self.gA1 = 0.066
        self.delA = 0.0015

        # Emissions parameters and Non-CO2 GHG with sigma = emissions/output
        self.gsigma1 = -0.015
        self.delgsig = 0.96
        self.asymgsig = -0.005
        self.e1 = 37.56
        self.miu1 = 0.05
        self.fosslim = 6000
        self.CumEmiss0 = 633.5

        # Climate damage parameters
        self.a1 = 0
        self.a2base = 0.003467
        self.a3 = 2.00

        # Abatement cost
        self.expcost2 = 2.6
        self.pback2050 = 515.0
        self.gback = -0.012
        self.cprice1 = 6
        self.gcprice = 0.025

        # Limits on emissions controls
        self.limmiu2070 = 1.0
        self.limmiu2120 = 1.1
        self.limmiu2200 = 1.05
        self.limmiu2300 = 1.0
        self.delmiumax = 0.12

        # Preferences, growth uncertainty, and timing
        self.betaclim = 0.5
        self.elasmu = 0.95
        self.prstp = 0.001
        self.pi = 0.05
        self.k0 = 295
        self.siggc1 = 0.01
        self.tstep = 5
        self.SRF = 1000000
        self.scale1 = 0.00891061
        self.scale2 = -6275.91

        # Additional parameters for non-CO2 GHGs
        self.eland0 = 5.9
        self.deland = 0.1
        self.F_Misc2020 = -0.054
        self.F_Misc2100 = 0.265
        self.F_GHGabate2020 = 0.518
        self.F_GHGabate2100 = 0.957
        self.ECO2eGHGB2020 = 9.96
        self.ECO2eGHGB2100 = 15.5
        self.emissrat2020 = 1.40
        self.emissrat2100 = 1.21
        self.Fcoef1 = 0.00955
        self.Fcoef2 = 0.861

        # Additional parameters for DFAIR model and climate equations
        self.emshare0 = 0.2173
        self.emshare1 = 0.224
        self.emshare2 = 0.2824
        self.emshare3 = 0.2763
        self.tau0 = 1000000
        self.tau1 = 394.4
        self.tau2 = 36.53
        self.tau3 = 4.304
        self.teq1 = 0.324
        self.teq2 = 0.44
        self.d1 = 236
        self.d2 = 4.07
        self.irf0 = 32.4
        self.irC = 0.019
        self.irT = 4.165
        self.fco22x = 3.93

        self.mat0 = 886.5128014
        self.res00 = 150.093
        self.res10 = 102.698
        self.res20 = 39.534
        self.res30 = 6.1865

        self.mateq = 588
        self.tbox10 = 0.1477
        self.tbox20 = 1.099454
        self.tatm0 = 1.24715

        #initialize_dynamic_parameters
        self.L = {1:self.pop1}  
        self.sig1 = self.e1 / (self.q1 * (1 - self.miu1))
        self.sigma = {1:self.sig1} 
        self.aL = {1:self.AL1}  
        self.gA = {}
        self.gsig = {}
        self.pbacktime = {}
        self.cpricebase = {}
        self.varpcc = {}
        self.rprecaut = {}
        self.RR1 = {}
        self.RR = {}
        self.miuup = {}

        #initialize_nonco2_parameters
        self.eland = {}
        self.CO2E_GHGabateB = {}
        self.F_Misc = {}
        self.emissrat = {}
        self.sigmatot = {}
        self.COST1TOT = {}

        # Initialize the initial conditions for FAIR model variables
        self.MAT = {1: self.mat0}
        self.TATM = {1: self.tatm0}
        self.RES0 = {1: self.res00}
        self.RES1 = {1: self.res10}
        self.RES2 = {1: self.res20}
        self.RES3 = {1: self.res30}
        self.TBOX1 = {1: self.tbox10}
        self.TBOX2 = {1: self.tbox20}
        self.alpha = {}
        self.IRFt = {}
        self.F_GHGabate = {1:self.F_GHGabate2020}
        self.CCATOT = {1: self.CumEmiss0}

        # Initialize the initial conditions for Emissions and Damages
        self.ECO2 = {}
        self.EIND = {}
        self.ECO2E = {}
        self.CACC = {}
        self.FORC = {}
        self.DAMFRAC = {}
        self.DAMAGES = {}
        self.ABATECOST = {}
        self.MCABATE = {}
        self.CPRICE = {}

        #Initialize the initial conditions for economic variables
        self.YGROSS = {}
        self.YNET = {}
        self.Y = {}
        self.C = {}
        self.CPC = {}
        self.I = {}
        self.S = {}
        self.K = {1: self.k0}  
        self.MIU = {}
        self.RFACTLONG = {1:1000000}
        self.RLONG = {}
        self.RSHORT = {}

        #Initialize the initial conditions for Utility
        self.PERIODU = {}
        self.TOTPERIODU = {}
        self.UTILITY = 0

    def initialize_dynamic_parameters(self):


        # Time preference for climate investments and precautionary effect
        self.rartp = exp(self.prstp + self.betaclim * self.pi) - 1

        # Initialize dynamic parameters for each time period
        for t in self.timesteps:
            # Growth rate of productivity (gA)
            self.gA[t] = self.gA1 * exp(-self.delA * 5 * (t - 1))

            # Change in sigma (rate of decarbonization) (gsig)
            self.gsig[t] = min(self.gsigma1 * (self.delgsig ** (t - 1)), self.asymgsig)

            # Population and labor (L)
            self.L[t] = self.pop1 if t == 1 else self.L[t-1] * ((self.popasym / self.L[t-1]) ** self.popadj)

            # Total factor productivity (aL)
            self.aL[t] = self.AL1 if t == 1 else self.aL[t-1] / (1 - self.gA[t])

            # CO2-emissions output ratio (sigma)
            self.sigma[t] = self.e1 / (self.q1 * (1 - self.miu1)) if t == 1 else self.sigma[t-1] * exp(5 * self.gsig[t])

            # Backstop price (pbacktime)
            self.pbacktime[t] = self.calculate_pbacktime(t)

            # Carbon price in base case (cpricebase)
            self.cpricebase[t] = self.cprice1 * ((1 + self.gcprice) ** (5 * (t - 1)))

            # Optimal long-run savings rate (optlrsav)
            self.optlrsav = (self.dk + 0.004) / (self.dk + 0.004 * self.elasmu + self.rartp) * self.gama

            # Precautionary dynamic parameters
            self.varpcc[t] = min(self.siggc1 ** 2 * 5 * (t - 1), self.siggc1 ** 2 * 5 * 47)
            self.rprecaut[t] = -0.5 * self.varpcc[t] * self.elasmu ** 2
            self.RR1[t] = 1 / ((1 + self.rartp) ** (self.tstep * (t - 1)))
            self.RR[t] = self.RR1[t] * (1 + self.rprecaut[t]) ** (-self.tstep * (t - 1))

            # Emissions limits (miuup)
            self.miuup[t] = self.calculate_miuup(t)

            # Other dynamic parameters
            # ...

    def initialize_nonco2_parameters(self):

        for t in self.timesteps:
            self.eland[t] = self.eland0 * ((1 - self.deland) ** (t - 1))
            self.CO2E_GHGabateB[t] = self.calculate_CO2E_GHGabateB(t)
            self.F_Misc[t] = self.F_Misc2020 + ((self.F_Misc2100 - self.F_Misc2020) / 16) * (t - 1) if t <= 16 else self.F_Misc2100
            self.emissrat[t] = self.emissrat2020 + ((self.emissrat2100 - self.emissrat2020) / 16) * (t - 1) if t <= 16 else self.emissrat2100
            self.sigmatot[t] = self.sigma[t] * self.emissrat[t]
            self.COST1TOT[t] = self.pbacktime[t] * self.sigmatot[t] / self.expcost2 / 1000

    def calculate_pbacktime(self, t):
        if t <= 7:
            return self.pback2050 * exp(-5 * 0.01 * (t - 7))
        else:
            return self.pback2050 * exp(-5 * 0.001 * (t - 7))

    def calculate_CO2E_GHGabateB(self, t):
        if t <= 16:
            return self.ECO2eGHGB2020 + ((self.ECO2eGHGB2100 - self.ECO2eGHGB2020) / 16) * (t - 1)
        else:
            return self.ECO2eGHGB2100 - self.ECO2eGHGB2020
        
    def calculate_miuup(self, t):
        if t == 1:
            return 0.05
        elif t == 2:
            return 0.10
        elif t > 57:
            return self.limmiu2300
        elif t > 37:
            return self.limmiu2200
        elif t > 20:
            return self.limmiu2120
        elif t > 11:
            return self.limmiu2070
        elif t > 8:
            return 0.85 + 0.05 * (t - 8)
        else:
            return self.delmiumax * (t - 1)
        
    def update_model(self, t):

        # Calculating gross world product (Gross output)
        self.YGROSS[t] = max((self.aL[t] * ((self.L[t] / 1000) ** (1 - self.gama))) * (self.K[t] ** self.gama), 0)

        # Total industrial emissions
        self.ECO2[t] = (self.sigma[t] * self.YGROSS[t] + self.eland[t]) * (1 - self.MIU[t])
        # Industrial emissions excluding deforestation
        self.EIND[t] = (self.sigma[t] * self.YGROSS[t]) * (1 - self.MIU[t])
        # Total CO2 emissions including emissions from land use change and GHG abatement
        self.ECO2E[t] = (self.sigma[t] * self.YGROSS[t] + self.eland[t] + self.CO2E_GHGabateB[t]) * (1 - self.MIU[t])

        # Reservoir law of motions
        if t == 1:
            # Use initial conditions for the first time step
            self.RES0[t] = self.res00
            self.RES1[t] = self.res10
            self.RES2[t] = self.res20
            self.RES3[t] = self.res30
        else:
            # For subsequent time steps, use the existing formulas
            self.RES0[t] = (self.emshare0 * self.tau0 * self.alpha[t] * (self.ECO2[t] / 3.667)) * (1 - exp(-self.tstep / (self.tau0 * self.alpha[t]))) + self.RES0[t - 1] * exp(-self.tstep / (self.tau0 * self.alpha[t]))
            self.RES1[t] = (self.emshare1 * self.tau1 * self.alpha[t] * (self.ECO2[t] / 3.667)) * (1 - exp(-self.tstep / (self.tau1 * self.alpha[t]))) + self.RES1[t - 1] * exp(-self.tstep / (self.tau1 * self.alpha[t]))
            self.RES2[t] = (self.emshare2 * self.tau2 * self.alpha[t] * (self.ECO2[t] / 3.667)) * (1 - exp(-self.tstep / (self.tau2 * self.alpha[t]))) + self.RES2[t - 1] * exp(-self.tstep / (self.tau2 * self.alpha[t]))
            self.RES3[t] = (self.emshare3 * self.tau3 * self.alpha[t] * (self.ECO2[t] / 3.667)) * (1 - exp(-self.tstep / (self.tau3 * self.alpha[t]))) + self.RES3[t - 1] * exp(-self.tstep / (self.tau3 * self.alpha[t]))

        # Atmospheric concentration equation
        if t == 1:
            self.MAT[t] = self.mat0
        else:
             self.MAT[t] = self.mateq + self.RES0[t] + self.RES1[t] + self.RES2[t] + self.RES3[t]
             self.MAT[t] = max(self.MAT[t], 10)

        # Accumulated Carbon
        if t < max(self.timesteps):
            self.CCATOT[t + 1] = self.CCATOT[t] + self.ECO2[t] * (5 / 3.666)
        self.CACC[t] = self.CCATOT[t] - (self.MAT[t] - self.mateq)

        # Radiative forcing equation
        if t < max(self.timesteps):
            self.F_GHGabate[t + 1] = self.Fcoef2 * self.F_GHGabate[t] + \
                                 self.Fcoef1 * self.CO2E_GHGabateB[t] * (1 - self.MIU[t])
        self.FORC[t] = self.fco22x * ((log(self.MAT[t] / self.mateq)) / log(2)) + self.F_Misc[t] + self.F_GHGabate[t]

        # Temperature equations
        if t == 1:
            # Use initial conditions for the first time step
            self.TBOX1[t] = self.tbox10
            self.TBOX2[t] = self.tbox20
            self.TATM[t] = self.tatm0
        else:
            # For subsequent time steps, use the existing formulas
            self.TBOX1[t] = self.TBOX1[t-1] * exp(-self.tstep / self.d1) + self.teq1 * self.FORC[t] * (1 - exp(-self.tstep / self.d1))
            self.TBOX2[t] = self.TBOX2[t-1] * exp(-self.tstep / self.d2) + self.teq2 * self.FORC[t] * (1 - exp(-self.tstep / self.d2))
            self.TATM[t] = max(self.TBOX1[t] + self.TBOX2[t], 0)
            self.TATM[t] = min(max(self.TATM[t], 0.5), 20)

        # Calculating damage fraction
        self.DAMFRAC[t] = (self.a1 * self.TATM[t]) + (self.a2base * (self.TATM[t] ** self.a3))
        # Calculating total damages
        self.DAMAGES[t] = self.YGROSS[t] * self.DAMFRAC[t]
        
        self.YNET[t] = max(self.YGROSS[t] * (1 - self.DAMFRAC[t]), 0)

        # Abatement costs
        self.ABATECOST[t] = self.YGROSS[t] * self.COST1TOT[t] * (self.MIU[t] ** self.expcost2)
        self.Y[t] = max(self.YNET[t] - self.ABATECOST[t], 0)
        self.I[t] = max(self.S[t] * self.Y[t], 0)
        self.C[t] = self.Y[t] - self.I[t]
        self.C[t] = max(self.C[t], 2)

        # Per capita consumption
        self.CPC[t] = 1000 * self.C[t] / self.L[t]
        self.CPC[t] = max(self.CPC[t], 0.01)

        self.MCABATE[t] = self.pbacktime[t] * (self.MIU[t] ** (self.expcost2 - 1))
        self.CPRICE[t] = self.pbacktime[t] * (self.MIU[t] ** (self.expcost2 - 1))
        
        # Update long-term factors
        if t == 1:
            self.RFACTLONG[t] = 1000000
        else:
            self.RFACTLONG[t] = self.SRF * (self.CPC[t] / self.CPC[1]) ** (-self.elasmu) * self.RR[t]
            self.RFACTLONG[t] = max(self.RFACTLONG[t], 0.0001)

            self.RLONG[t] = -log(self.RFACTLONG[t] / self.SRF) / (5 * t)
            self.RSHORT[t] = -log(self.RFACTLONG[t] / self.RFACTLONG[t]) / 5

        # IRF equation
        self.IRFt[t] = max(self.irf0 + self.irC * self.CACC[t] + self.irT * self.TATM[t], 0)

        # Utility
        self.PERIODU[t] = ((self.C[t] * 1000 / self.L[t]) ** (1 - self.elasmu) - 1) / (1 - self.elasmu) - 1
        self.TOTPERIODU[t] = self.PERIODU[t] * self.L[t] * self.RR[t]

    def compute_utility(self):
        self.UTILITY = self.tstep * self.scale1 * sum(self.TOTPERIODU[t] for t in self.timesteps) + self.scale2
        return self.UTILITY
    
class DICEOptimizerIPM:
    def __init__(self, dice_model):
        self.model = dice_model
        self.num_timesteps = len(dice_model.timesteps)

    def objective_function(self, decision_variables):
        # Unpack decision variables
        S_values = decision_variables[:self.num_timesteps]
        K_values = decision_variables[self.num_timesteps:2*self.num_timesteps]
        MIU_values = decision_variables[2*self.num_timesteps:3*self.num_timesteps]
        ALPHA_values = decision_variables[3*self.num_timesteps:]

        # Update the model with new values
        for t in self.model.timesteps:
            if t == 1:
                self.model.MIU[t] = min(MIU_values[t - 1], self.model.miuup[t])
                self.model.S[t] = max(S_values[t - 1], 0) 
                self.model.alpha[t] = min(max(ALPHA_values[t - 1], 0.1), 100)
                self.model.K[t] = self.model.k0
            else :
                self.model.MIU[t] = min(MIU_values[t - 1], self.model.miuup[t])
                self.model.S[t] = max(S_values[t - 1], 0) if t <= 37 else 0.28
                self.model.alpha[t] = min(max(ALPHA_values[t - 1], 0.1), 100)
                self.model.K[t] = max(K_values[t - 1], 1)

            self.model.update_model(t)

        return -dice_model.compute_utility()

    def optimize(self):
        # Initial guess for decision variables
        initial_guess = np.full(4 * self.num_timesteps, 0.5)
        bounds = Bounds([0]*4*self.num_timesteps, [1, float('inf'), 1, float('inf')]*self.num_timesteps)

        # Run the optimization
        result = minimize(self.objective_function, initial_guess, method='trust-constr', bounds=bounds)

        return result
    
class DICEOptimizerSQP:
    def __init__(self, dice_model):
        self.model = dice_model
        self.num_timesteps = len(dice_model.timesteps)
    
    def objective_function(self, decision_variables):
        # Unpack decision variables
        MIU_values = decision_variables[:self.num_timesteps]
        S_values = decision_variables[self.num_timesteps:2*self.num_timesteps]
        alpha_values = decision_variables[2*self.num_timesteps:3*self.num_timesteps]
        K_values = decision_variables[3*self.num_timesteps:]

        penalty = 0
        penalty_coefficient = 10000  # 惩罚系数

        for t in self.model.timesteps:
            if t == 1:
                self.model.MIU[t] = min(MIU_values[t - 1], self.model.miuup[t])
                self.model.S[t] = S_values[t - 1] if t <= 37 else 0.28
                self.model.alpha[t] = min(max(alpha_values[t - 1], 0.1), 100)
                self.model.K[t] = self.model.k0
            else :
                self.model.MIU[t] = min(MIU_values[t - 1], self.model.miuup[t])
                self.model.S[t] = S_values[t - 1] if t <= 37 else 0.28
                self.model.alpha[t] = min(max(alpha_values[t - 1], 0.1), 100)
                self.model.K[t] = max(K_values[t - 1], 1)


            # Check and apply the constraint on K[t]
                K_calculated = (1 - self.model.dk) ** self.model.tstep * self.model.K[t-1] + self.model.tstep * self.model.I[t-1]
                if self.model.K[t] > K_calculated:
                    penalty += penalty_coefficient * (self.model.K[t] - K_calculated)

            self.model.update_model(t)

        return -dice_model.compute_utility()

    def optimize(self):
        # Initial guess for decision variables
        initial_guess = np.zeros(4 * len(dice_model.timesteps))  
        # Run the optimization
        result = minimize(self.objective_function, initial_guess, method='SLSQP')
        
        return result

class DICEOptimizerSA:
    def __init__(self, dice_model):
        self.model = dice_model
        self.num_timesteps = len(dice_model.timesteps)
        self.size_of_decision_variables = self.num_timesteps * 4

    def objective_function(self, decision_variables):
        # Unpack decision variables
        S_values = decision_variables[:self.num_timesteps]
        K_values = decision_variables[self.num_timesteps:2*self.num_timesteps]
        MIU_values = decision_variables[2*self.num_timesteps:3*self.num_timesteps]
        ALPHA_values = decision_variables[3*self.num_timesteps:]

        for t in self.model.timesteps:
            if t == 1:
                self.model.MIU[t] = min(MIU_values[t - 1], self.model.miuup[t])
                self.model.S[t] = max(S_values[t - 1], 0) 
                self.model.alpha[t] = min(max(ALPHA_values[t - 1], 0.1), 100)
                self.model.K[t] = self.model.k0
            else :
                self.model.MIU[t] = min(MIU_values[t - 1], self.model.miuup[t])
                self.model.S[t] = max(S_values[t - 1], 0) if t <= 37 else 0.28
                self.model.alpha[t] = min(max(ALPHA_values[t - 1], 0.1), 100)
                self.model.K[t] = max(K_values[t - 1], 1)

            self.model.update_model(t)

        return -dice_model.compute_utility()


    def optimize(self):
        # Bounds for decision variables
        large_value = 10000000  
        bounds = [(0, 2)] * self.num_timesteps + [(0, 1)] * self.num_timesteps + \
                [(0, 100)] * self.num_timesteps + [(0, large_value)] * self.num_timesteps

        result = dual_annealing(self.objective_function, bounds=bounds)

        return result.x
    
if __name__ == '__main__':
    # Initialize DICE model and optimizer
    dice_model = DICEModel()
    dice_model.initialize_dynamic_parameters()
    dice_model.initialize_nonco2_parameters()

    #optimizer = DICEOptimizerSQP(dice_model)
    #optimizer = DICEOptimizerSA(dice_model)
    optimizer = DICEOptimizerIPM(dice_model)
    result = optimizer.optimize()

    # List of variables to be exported
    variables_to_export = [ 
        'MAT', 'TATM', 'RES0', 'RES1', 'RES2', 
        'RES3', 'TBOX1', 'TBOX2', 'alpha', 'F_GHGabate', 'CCATOT', 
        'FORC', 'DAMFRAC', 'ABATECOST', 
        'YGROSS', 'YNET', 'Y', 'C', 'CPC', 'I', 'S', 'K', 
        'MIU', 'PERIODU', 'TOTPERIODU', 'CPRICE', 'UTILITY'
    ]

    # Create an empty DataFrame to store all data
    combined_df = pd.DataFrame()

    for var_name in variables_to_export:
        var_data = getattr(dice_model, var_name)

        # Convert data to DataFrame
        if isinstance(var_data, dict):
            # For variables that are of type dictionary
            temp_df = pd.DataFrame(list(var_data.items()), columns=['Time', var_name])
        else:
            # For variables that are single numeric values
            temp_df = pd.DataFrame({var_name: [var_data]})
            # Add a time column
            temp_df['Time'] = 1

        # Initialize combined_df with the first variable
        if combined_df.empty:
            combined_df = temp_df
        else:
            # Merge into combined_df based on the 'Time' column
            combined_df = pd.merge(combined_df, temp_df, on='Time', how='outer')

    # Write the combined DataFrame to an Excel file
    combined_df.to_excel(r'DICE_Model_Combined_Results.xlsx', index=False)