import numpy as np
import pandas as pd
import sympy as sp
from IPython.display import display, Markdown
from scipy.constants import convert_temperature as conv_temp

from Classes.LED_params import LEDparams

class Model:
    def __init__(self, LEDparams:LEDparams, x0:np.ndarray, T_amb0:float=conv_temp(25.0, 'C', 'K')) -> None:
        """
        x0 : Initial state

        u1: current through the peltier module
        u2: PWM fan

        x1: SoC
        x2: T_HP_c
        x3: T_HP_h
        """
        # State input initialization
        self.x = x0
        self.u = np.array([0.0, 0.0])

        # Parameters initialization
        self._init_params(LEDparams, T_amb0)

        # Dictionary to connect symbolic variables with values
        self.params_values = {
            'n_BT':         self.n_BT, # Battery
            'Q_max':        self.Q_BT_max,
            'R_in':         self.R_BT_int,
            'P_rest':       self.P_el_rest,
            'a3':           self.BT_coefs["a3"].iloc[0],
            'a2':           self.BT_coefs["a2"].iloc[0],
            'a1':           self.BT_coefs["a1"].iloc[0],
            'a0':           self.BT_coefs["a0"].iloc[0],
            'U_FAN':        self.U_FAN, # Fan
            'I_FAN':        self.I_FAN,
            'y_FAN':        self.FAN_coefs["y_min"].iloc[0],
            'a_FAN':        self.FAN_coefs["a"].iloc[0],
            'b_FAN':        self.FAN_coefs["b"].iloc[0],
            'k_FAN':        self.FAN_coefs["k"].iloc[0],
            'I_LED':        self.I_LED, # LED
            'x_LED':        self.x_LED,
            'P_r':          self.P_LED_r,
            'S_M':          self.S_M, # HP
            'R_M':          self.R_M,
            'K_M':          self.K_M,
            'cp_Al':        self.cp_Al, # Thermal
            'cp_w':         self.cp_H2O,
            'T_amb':        self.T_amb,
            'm_c':          self.m_2 + self.m_4,
            'm_h':          self.m_1,
            'R_floor':      self.R_floor_lambda,
            'R4_lambda':    self.R_4_lambda,
            'R5':           self.R_5,
        }

        # Operational point initialization
        self._init_operational_point()

        # Symbolic initialization
        self._init_sym_model()

    def _init_params(self, LEDparams:LEDparams, T_amb0:float) -> None:
        self.P_el_rest = 1.0 # W TODO get better value
        self.n_BT = 2
        self.Q_BT_max = 3.0 * 3600 # As (Ah = 3600 As)
        self.R_BT_int = 0.1 # Ohm
        self.BT_coefs = pd.read_csv(
            'C:\\Users\\giaco\\Git_Repositories\\Semester_Thesis_1\\Data\\Battery\\battery_fitted_coefficients_3rd.csv'
            )

        # Fan parameters
        self.I_FAN = 0.13 # A
        self.U_FAN = 12.0 # V
        self.FAN_coefs = pd.read_csv(
            'C:\\Users\\giaco\\Git_Repositories\\Semester_Thesis_1\\Data\\Fan\\fan_coefficients.csv'
            )

        # LED parameters
        self.I_LED = LEDparams.I_LED # A
        self.x_LED = LEDparams.x_LED_tot # duty cycle
        self.P_LED_r = LEDparams.P_r # W

        # Thermal parameters
        self.T_amb = T_amb0 # K
        self.cp_Al = 897.0 # J/kgK https://en.wikipedia.org/wiki/6061_aluminium_alloy
        self.cp_H2O = 4180.0 # J/kgK https://www.engineeringtoolbox.com/specific-heat-capacity-water-d_660.html
        # self.cp_air = 1006.0 # J/kgK https://www.engineeringtoolbox.com/air-specific-heat-capacity-d_705.html?vA=37&degree=C&pressure=1bar#

        # Top thermal parameters - Diffuser
        # TODO estimation of parameters
        self.R_4_lambda = 0.5 # K/W
        self.R_5 = 25.0 # K/W

        # Top Al thermal parameters
        self.m_2 = 0.0638 # kg
        self.m_4 = 0.0642 # kg

        # Bottom Al thermal parameters - Heat sink
        self.m_1 = 0.0876 # kg
        self.R_floor_lambda = 2.6 # K/W # TODO estimated real time

        # Heat pump - peltier module
        HP_params = pd.read_csv(
            'C:\\Users\\giaco\\Git_Repositories\\Semester_Thesis_1\\Data\\Heat Pump\\HP_fitted_coefficients.csv'
            )
        self.R_M = HP_params["R_M"].iloc[0] # Ohm
        self.S_M = HP_params["S_M"].iloc[0] # V/K
        self.K_M = HP_params["K_M"].iloc[0] # W/K

        I_HP_max_datasheet = HP_params["I_max"].iloc[0] # A
        I_HP_max_electronics = 3.0 # A when attached to the battery
        self.I_HP_max = min(I_HP_max_datasheet, I_HP_max_electronics) # A

    def _init_operational_point(self) -> None:
        x_SoC = 0.85
        T_c   = conv_temp(self.T_amb, 'C', 'K') # K
        T_h   = conv_temp(self.T_amb + 30.0, 'C', 'K') # K
        I_HP  = 1.2 # A
        x_FAN = 1.0 # duty cycle

        self.x_op = np.array([x_SoC, T_c, T_h])
        self.u_op = np.array([I_HP, x_FAN])

    def _init_sym_model(self) -> None:
        # State variables
        x_SoC, T_c, T_h = sp.symbols('x_SoC, T_c, T_h')
        self.sym_x      = sp.Matrix([x_SoC, T_c, T_h])

        # Input variables
        I_HP, x_FAN = sp.symbols('I_HP, x_FAN')
        self.sym_u  = sp.Matrix([I_HP, x_FAN])

        # Battery parameters
        n, Q_max, R_in, P_rest = sp.symbols('n_BT, Q_max, R_in, P_rest')
        a3, a2, a1, a0 = sp.symbols('a3, a2, a1, a0')
        U_oc = a3 * x_SoC**3 + a2 * x_SoC**2 + a1 * x_SoC + a0 # V
        # display(Markdown(r"$U_{oc}(x_{SoC}):$"), U_oc.subs(self.params_values))

        # Fan parameters
        U_FAN, I_FAN = sp.symbols('U_FAN, I_FAN')
        y_min, a, b, k = sp.symbols('y_FAN, a_FAN, b_FAN, k_FAN')
        R_air_alpha = y_min + (1 / (x_FAN + a) + b - y_min) / (1 + sp.exp(-k * x_FAN)) # K/W
        # display(Markdown(r"$R_{air}^\alpha(x_{FAN}):$"), R_air_alpha)
        # display(Markdown(r"$R_{air}^\alpha(x_{FAN}=1):$"), R_air_alpha.subs(self.params_values).subs({x_FAN: 1.0}))
        # display(Markdown(r"$R_{air}^\alpha(x_{FAN}=0):$"), R_air_alpha.subs(self.params_values).subs({x_FAN: 0.0}))

        # LED parameters
        I_LED, x_LED, P_r = sp.symbols('I_LED, x_LED, P_r')

        # HP parameters - Peltier module
        S_M, R_M, K_M = sp.symbols('S_M, R_M, K_M')
        U_HP = S_M * (T_h - T_c) + R_M * I_HP # V
        Q_c = S_M * I_HP * T_c - 0.5 * R_M * I_HP**2 - K_M * (T_h - T_c) # W

        # Thermal parameters
        cp_Al, T_amb = sp.symbols('cp_Al, T_amb')

        # Top thermal parameters - Diffuser
        R4_lambda, R5 = sp.symbols('R4_lambda, R5')
        Q_LEDcell = (T_c - T_amb) / (R5 + R4_lambda) # W

        # Top Al thermal parameters - Diffuser
        m_c = sp.symbols('m_c')

        # Bottom Al thermal parameters - Heat sink
        m_h, R_floor_lambda = sp.symbols('m_h, R_floor')
        R_eq = (R_floor_lambda * R_air_alpha) / (R_floor_lambda + R_air_alpha) # K/W
        # display(Markdown(r"$R_{eq}(x_{FAN}=1):$"), R_eq.subs(self.params_values).subs({x_FAN: 1.0}))
        # display(Markdown(r"$R_{eq}(x_{FAN}=0):$"), R_eq.subs(self.params_values).subs({x_FAN: 0.0}))

        ### Calculations
        # Output: T_cell
        T_cell = R5 * Q_LEDcell + T_amb # K
        # display(Markdown(r"$T_{cell}:$"), T_cell.subs(self.params_values))

        # Output: I_BT
        # I_BT = U_oc + R_in * I_HP + R_in * I_LED * x_LED - sp.sqrt(U_oc**2 - 2 * R_in * I_LED * x_LED * U_oc - 2 * R_in * U_oc * I_HP + R_in**2 * I_HP**2 + 2 * R_in * I_LED * x_LED * R_in * I_HP - 4 * R_in * I_FAN * U_FAN * x_FAN - 4 * R_in * P_rest + (R_in * I_LED * x_LED)**2) # A
        I_BT = U_oc + R_in * sp.sqrt(I_HP**2) + R_in * I_LED * x_LED - sp.sqrt(U_oc**2 - 2 * R_in * I_LED * x_LED * U_oc - 2 * R_in * U_oc * sp.sqrt(I_HP**2) + R_in**2 * I_HP**2 + 2 * R_in * I_LED * x_LED * R_in * sp.sqrt(I_HP**2) - 4 * R_in * I_FAN * U_FAN * x_FAN - 4 * R_in * P_rest + (R_in * I_LED * x_LED)**2) # A
        U_BT = U_oc - R_in * I_BT # V
        # display(Markdown(r"$U_{BT}:$"), U_BT.subs(self.params_values))

        # HP calculation
        P_HP = U_HP * I_HP # W
        COP = 1 + Q_c / P_HP

        # LED calculation
        P_LED = I_LED * U_BT * x_LED # W
        Q_LED = P_LED - P_r # W

        # Nonlinear ODEs
        dTh_dt = (1 / (m_h * cp_Al)) * (Q_c + P_HP - (T_h - T_amb) / R_eq)
        dTc_dt = (1 / (m_c * cp_Al)) * (Q_LED - Q_LEDcell - Q_c)
        dxSoC_dt = - I_BT / (n * Q_max)

        # Symbolic dynamics, output and linearization
        self.f_symb = sp.Matrix([dxSoC_dt, dTc_dt, dTh_dt])
        self.g_symb = sp.Matrix([T_c, I_BT])
        # display(Markdown(r"$\dot{x} = f(x, u):$"), self.f_symb)
        # display(Markdown(r"$y = g(x, u):$"), self.g_symb)

        self.A_symb = self.f_symb.jacobian(self.sym_x)
        self.B_symb = self.f_symb.jacobian(self.sym_u)
        self.C_symb = self.g_symb.jacobian(self.sym_x)
        self.D_symb = self.g_symb.jacobian(self.sym_u)
        # display(Markdown(r"$A = \nabla_x f:$"), self.A_symb.subs(self.params_values).subs({x_SoC: 0.85, T_c: 25.0, T_h: 55.0, x_FAN: 1.0}))
        # display(Markdown(r"$B = \nabla_u f:$"), self.B_symb.subs(self.params_values).subs({x_SoC: 0.85, T_c: 25.0, T_h: 55.0, x_FAN: 1.0}))

        # Create numerical functions with parameters already inserted
        self.f_num = sp.lambdify((self.sym_x, self.sym_u), self.f_symb.subs(self.params_values), modules="numpy")
        self.g_num = sp.lambdify((self.sym_x, self.sym_u), self.g_symb.subs(self.params_values), modules="numpy")

        self.A_num = sp.lambdify((self.sym_x, self.sym_u), self.A_symb.subs(self.params_values), modules="numpy")
        self.B_num = sp.lambdify((self.sym_x, self.sym_u), self.B_symb.subs(self.params_values), modules="numpy")
        self.C_num = sp.lambdify((self.sym_x, self.sym_u), self.C_symb.subs(self.params_values), modules="numpy")
        self.D_num = sp.lambdify((self.sym_x, self.sym_u), self.D_symb.subs(self.params_values), modules="numpy")

        ## Values
        # Voltages
        self.U_BT_num = sp.lambdify((self.sym_x, self.sym_u), U_BT.subs(self.params_values), modules="numpy")
        self.U_oc_num = sp.lambdify((self.sym_x, self.sym_u), U_oc.subs(self.params_values), modules="numpy")
        self.U_HP_num = sp.lambdify((self.sym_x, self.sym_u), U_HP.subs(self.params_values), modules="numpy")
        self.COP_num  = sp.lambdify((self.sym_x, self.sym_u), COP.subs(self.params_values), modules="numpy")

    def get_linearization(self, xss:np.ndarray=None, uss:np.ndarray=None) -> np.ndarray:
        if xss is None:
            xss = self.x_op
        if uss is None:
            uss = self.u_op

        A = self.A_num(xss, uss)
        B = self.B_num(xss, uss)
        h = self.f_num(xss, uss)
        C = self.C_num(xss, uss)
        D = self.D_num(xss, uss)
        l = self.g_num(xss, uss)

        return np.array(A).astype(np.float32), np.array(B).astype(np.float32), np.array(h).astype(np.float32), np.array(C).astype(np.float32), np.array(D).astype(np.float32), np.array(l).astype(np.float32)
    
    def dynamics_f(self, x:np.ndarray, u:np.ndarray) -> np.ndarray:
        return np.array(self.f_num(x, u)).flatten()

    def observer_g(self, x:np.ndarray, u:np.ndarray) -> np.ndarray:
        return np.array(self.g_num(x, u)).flatten()

    def discretized_update(self, u:np.ndarray, dt:float) -> np.ndarray:
        """
        Update the states using Runge-Kutta of 4th order integration.
        """
        x = self.x

        # Bound input
        u = self._input_bounds(x, u)

        # Runge-Kutta 4th order
        k = np.zeros((4, len(x)))
        k[0] = self.dynamics_f(x, u)
        k[1] = self.dynamics_f(x + 0.5 * dt * k[0], u)
        k[2] = self.dynamics_f(x + 0.5 * dt * k[1], u)
        k[3] = self.dynamics_f(x + dt * k[2], u)

        self.x = x + (dt / 6.0) * (k[0] + 2 * k[1] + 2 * k[2] + k[3])
        self.u = u

        # Bound states
        self.x = self._states_bounds(self.x)
        
        return self.x
    
    def get_output(self) -> np.ndarray:
        return self.observer_g(self.x, self.u)
    
    def get_values(self, x:np.ndarray, u:np.ndarray) -> np.ndarray:
        U_BT = np.array(self.U_BT_num(x, u)).flatten()
        U_oc = np.array(self.U_oc_num(x, u)).flatten()
        U_HP = np.array(self.U_HP_num(x, u)).flatten()
        COP  = np.array(self.COP_num(x, u)).flatten()
        return np.array([U_BT, U_oc, U_HP, COP]).flatten()

    def _input_bounds(self, x:np.ndarray, u:np.ndarray) -> np.ndarray:
        # HP current bounds
        u[0] = np.clip(u[0], -self.I_HP_max, self.I_HP_max)

        I_HP_max_I_source = (self.U_BT_num(x, u) - self.S_M * (x[2] - x[1])) / self.R_M
        I_HP_min_I_source = (-self.U_BT_num(x, u) - self.S_M * (x[2] - x[1])) / self.R_M
        u[0] = np.clip(u[0], I_HP_min_I_source, I_HP_max_I_source)

        # Fan duty cycle bounds
        u[1] = np.clip(u[1], 0.0, 1.0)
        return u
    
    def _states_bounds(self, x:np.ndarray) -> np.ndarray:
        x[0] = np.clip(x[0], 0.0, 1.0)
        x[1] = np.clip(x[1], 0.0, conv_temp(100.0, 'C', 'K'))
        x[2] = np.clip(x[2], 0.0, conv_temp(100.0, 'C', 'K'))
        return x
