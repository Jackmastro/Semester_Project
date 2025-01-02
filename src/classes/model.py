import numpy as np
import pandas as pd
import sympy as sp
from IPython.display import display, Markdown
from scipy.constants import convert_temperature as conv_temp
from scipy.signal import cont2discrete

from classes.LED_params import LEDparams

from data import load_coefficients

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
        # Initial condition
        self.x0 = x0
        self.x_prev = x0
        self.x_next = None
        self.u = np.array([0.0, 0.0]) # needed for symbolic initialization

        # Parameters initialization
        self._init_params(LEDparams, T_amb0)

        # Dictionary to connect symbolic variables with values
        self.params_values = {
            'n_BT':         self.n_BT, # Battery
            'Q_max':        self.Q_BT_max,
            'R_in':         self.R_BT_int,
            'P_rest':       self.P_rest,
            'a3':           self.BT_coefs["a3"].iloc[0],
            'a2':           self.BT_coefs["a2"].iloc[0],
            'a1':           self.BT_coefs["a1"].iloc[0],
            'a0':           self.BT_coefs["a0"].iloc[0],
            'U_FAN':        self.U_FAN, # Fan
            'I_FAN':        self.I_FAN,
            'q_FAN':        self.FAN_lin_coefs["q"].iloc[0],
            'm_FAN':        self.FAN_lin_coefs["m"].iloc[0],
            'a_FAN':        self.FAN_coefs["a"].iloc[0],
            'b_FAN':        self.FAN_coefs["b"].iloc[0],
            'c_FAN':        self.FAN_coefs["c"].iloc[0],
            'd_FAN':        self.FAN_coefs["d"].iloc[0],
            'I_LED':        self.I_LED, # LED
            'x_LED':        self.x_LED_tot,
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
            'Q_rest':       self.Q_rest,
            'I_rest':       self.I_rest
        }

        # Operational point initialization
        self._init_operational_point()

        # Symbolic initialization
        self._init_sym_model()

    def _init_params(self, LEDparams:LEDparams, T_amb:float) -> None:
        # Battery parameters
        self.P_rest = 1.0 # W TODO get better value
        self.I_rest = 0.5 # A TODO get better value
        self.U_rest = self.P_rest / self.I_rest # V
        self.n_BT = 2
        self.Q_BT_max = 3.0 * 3600 # As (used conversion: Ah = 3600 As)
        self.R_BT_int = 0.1 # Ohm
        self.BT_coefs = load_coefficients('battery\\battery_fitted_coefficients_3rd.csv')
        self.I_BT_max = 15 # A
        self.I_BT_min = -3 # A

        # Fan parameters
        self.I_FAN = 0.13 # A
        self.U_FAN = 12.0 # V
        self.FAN_coefs = load_coefficients('fan\\fan_coefficients.csv')
        self.FAN_lin_coefs = load_coefficients('fan\\fan_linear_coefficients.csv')

        # LED parameters
        self.I_LED = LEDparams.I_LED # A
        self.x_LED_tot = LEDparams.x_LED_tot # total duty cycle
        self.P_LED_r = LEDparams.P_r # W

        # Thermal parameters
        self.T_amb = T_amb # K
        self.cp_Al = 897.0 # J/kgK https://en.wikipedia.org/wiki/6061_aluminium_alloy
        self.cp_H2O = 4180.0 # J/kgK https://www.engineeringtoolbox.com/specific-heat-capacity-water-d_660.html
        # self.cp_air = 1006.0 # J/kgK https://www.engineeringtoolbox.com/air-specific-heat-capacity-d_705.html?vA=37&degree=C&pressure=1bar#

        # Top thermal parameters - Diffuser
        # TODO estimation of parameters
        self.R_4_lambda = 5 # K/W
        self.R_5 = 25.0 # K/W

        # Top Al thermal parameters
        self.m_2 = 0.0638 # kg
        self.m_4 = 0.0642 # kg
        self.Q_rest = self.P_rest # W # TODO only valid when battery attached. Case when attached via cable

        # Bottom Al thermal parameters - Heat sink
        self.m_1 = 0.0876 # kg
        self.R_floor_lambda = 2.6 # K/W # TODO estimated real time

        # Heat pump - peltier module
        HP_params = load_coefficients('heat_pump\\HP_fitted_coefficients.csv')
        self.R_M = HP_params["R_M"].iloc[0] # Ohm
        self.S_M = HP_params["S_M"].iloc[0] # V/K
        self.K_M = HP_params["K_M"].iloc[0] # W/K

        self.DeltaT_max = HP_params["DeltaT_max"].iloc[0] # K

        I_HP_max_datasheet = HP_params["I_max"].iloc[0] # A
        I_HP_max_electronics = 3.0 # A when attached to the battery
        self.I_HP_max = min(I_HP_max_datasheet, I_HP_max_electronics) # A

    def _init_operational_point(self) -> None:
        x_SoC = 0.85
        T_c   = conv_temp(self.T_amb, 'C', 'K') # K
        T_h   = conv_temp(self.T_amb + 10.0, 'C', 'K') # K
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
        q, m = sp.symbols('q_FAN, m_FAN')
        R_air_alpha = q + m * x_FAN # K/W
        # a, b, c, d = sp.symbols('a_FAN, b_FAN, c_FAN, d_FAN')
        # R_air_alpha = c + (1 / (x_FAN + a) + b - c) / (1 + sp.exp(-d * x_FAN)) # K/W
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
        m_h, R_floor_lambda, Q_rest = sp.symbols('m_h, R_floor, Q_rest')
        R_eq = (R_floor_lambda * R_air_alpha) / (R_floor_lambda + R_air_alpha) # K/W
        # display(Markdown(r"$R_{eq}(x_{FAN}=1):$"), R_eq.subs(self.params_values).subs({x_FAN: 1.0}))
        # display(Markdown(r"$R_{eq}(x_{FAN}=0):$"), R_eq.subs(self.params_values).subs({x_FAN: 0.0}))

        ### Calculations
        # FAN calculation
        P_FAN = I_FAN * U_FAN * x_FAN # W

        # HP calculation
        P_HP = U_HP * I_HP # W
        COP = 1 + Q_c / P_HP # only valid for positive I_HP (cooling)

        # BT calculation
        I_BT = (U_oc + R_in*I_LED*x_LED - sp.sqrt(U_oc**2 - 2*R_in*I_LED*x_LED*U_oc + (R_in*I_LED*x_LED)**2 - 4*R_in*(P_rest + P_FAN + sp.sqrt(P_HP**2)))) / (2*R_in) # A
        U_BT = U_oc - R_in * I_BT # V
        P_BT = U_BT * I_BT # W
        # display(Markdown(r"$U_{BT}:$"), U_BT.subs(self.params_values))
        # display(Markdown(r"$U_{BT}^{}:$"), U_BT.subs(self.params_values).subs({I_HP:3.0,  x_FAN: 1.0, x_SoC: 1.0}))
        # display(Markdown(r"$U_{BT}^{max}:$"), U_BT.subs(self.params_values).subs({I_HP:3.0,  x_FAN: 0.0, x_SoC: 1.0}))
        # display(Markdown(r"$U_{BT}^{min}:$"), U_BT.subs(self.params_values).subs({I_HP:3.0,  x_FAN: 0.0, x_SoC: 0.0}))
        # display(Markdown(r"$U_{BT}^{}:$"), U_BT.subs(self.params_values).subs({I_HP:-3.0, x_FAN: 1.0, x_SoC: 1.0}))
        # display(Markdown(r"$U_{BT}^{max}:$"), U_BT.subs(self.params_values).subs({I_HP:-3.0, x_FAN: 0.0, x_SoC: 1.0}))
        # display(Markdown(r"$U_{BT}^{min}:$"), U_BT.subs(self.params_values).subs({I_HP:-3.0, x_FAN: 0.0, x_SoC: 0.0}))

        # LED calculation
        P_LED = I_LED * U_BT * x_LED # W
        Q_LED = P_LED - P_r # W

        # Output: T_cell
        T_cell = R5 * Q_LEDcell + T_amb # K
        # display(Markdown(r"$T_{cell}:$"), T_cell)

        # Nonlinear ODEs
        dTh_dt = (1 / (m_h * cp_Al)) * (Q_c + P_HP - (T_h - T_amb) / R_eq + Q_rest)
        dTc_dt = (1 / (m_c * cp_Al)) * (Q_LED - Q_LEDcell - Q_c)
        dxSoC_dt = - I_BT / (n * Q_max)

        # Symbolic dynamics, output and linearization
        self.f_symb = sp.Matrix([dxSoC_dt, dTc_dt, dTh_dt])
        self.g_symb = sp.Matrix([T_c])
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

        # Values
        self.U_BT_num   = sp.lambdify((self.sym_x, self.sym_u), U_BT.subs(self.params_values), modules="numpy")
        self.U_oc_num   = sp.lambdify((self.sym_x, self.sym_u), U_oc.subs(self.params_values), modules="numpy")
        self.U_HP_num   = sp.lambdify((self.sym_x, self.sym_u), U_HP.subs(self.params_values), modules="numpy")
        self.COP_num    = sp.lambdify((self.sym_x, self.sym_u), COP.subs(self.params_values), modules="numpy")
        self.I_BT_num   = sp.lambdify((self.sym_x, self.sym_u), I_BT.subs(self.params_values), modules="numpy")
        self.T_cell_num = sp.lambdify((self.sym_x, self.sym_u), T_cell.subs(self.params_values), modules="numpy")

    def get_continuous_linearization(self, xss:np.ndarray=None, uss:np.ndarray=None) -> np.ndarray:
        if xss is None:
            xss = self.x_op
        if uss is None:
            uss = self.u_op

        A = self.A_num(xss, uss)
        B = self.B_num(xss, uss)
        h = self.f_num(xss, uss).reshape(-1,) - A @ xss - B @ uss
        C = self.C_num(xss, uss)
        D = self.D_num(xss, uss)
        l = self.g_num(xss, uss).reshape(-1,) - C @ xss - D @ uss

        return np.array(A).astype(np.float32), np.array(B).astype(np.float32), np.array(h).astype(np.float32), np.array(C).astype(np.float32), np.array(D).astype(np.float32), np.array(l).astype(np.float32)
    
    def get_discrete_linearization(self, Ts:float, xss:np.ndarray=None, uss:np.ndarray=None) -> np.ndarray:
        A, B, h, C, D, l = self.get_continuous_linearization(xss, uss)
        A_d, B_d, C_d, D_d, _ = cont2discrete((A, B, C, D), dt=Ts, method='zoh')
        
        return A_d, B_d, h, C_d, D_d, l
    
    def dynamics_f(self, x:np.ndarray, u:np.ndarray) -> np.ndarray:
        return np.array(self.f_num(x, u)).flatten()

    def observer_g(self, x:np.ndarray, u:np.ndarray) -> np.ndarray:
        return np.array(self.g_num(x, u)).flatten()

    def discretized_update(self, u:np.ndarray, dt:float) -> np.ndarray:
        """
        Update the states using Runge-Kutta of 4th order integration.
        """
        x = self.x_prev

        # Bound input
        u_bounded = self._input_bounds(x, u)

        # Runge-Kutta 4th order
        k = np.zeros((4, len(x)))
        k[0] = self.dynamics_f(x, u_bounded)
        k[1] = self.dynamics_f(x + 0.5 * dt * k[0], u_bounded)
        k[2] = self.dynamics_f(x + 0.5 * dt * k[1], u_bounded)
        k[3] = self.dynamics_f(x + dt * k[2], u_bounded)

        self.x_next = x + (dt / 6.0) * (k[0] + 2 * k[1] + 2 * k[2] + k[3])

        # Bound states
        self.x_next = self._states_bounds(self.x_next)

        # Update
        self.u = u_bounded
        self.x_prev = self.x_next
        
        return self.x_next, self.u
    
    def get_output(self) -> np.ndarray:
        return self.observer_g(self.x_prev, self.u)
    
    def get_values(self, x:np.ndarray, u:np.ndarray) -> dict:
        return {
            'U_BT':     self.U_BT_num(x, u),
            'U_oc':     self.U_oc_num(x, u),
            'U_HP':     self.U_HP_num(x, u),
            'COP':      self.COP_num(x, u),
            'I_BT':     self.I_BT_num(x, u),
            'T_cell':   self.T_cell_num(x, u),
        }
    
    def get_constraints_U_BT2I_HP(self, delta_T:np.ndarray) -> np.ndarray:
        # Zero order approximation for U_BT
        U_BT = self.U_BT_num(self.x_op, self.u_op) # TODO use max and min values
        I_HP_max_I_source = (U_BT - self.S_M * delta_T) / self.R_M
        I_HP_min_I_source = (- U_BT - self.S_M * delta_T) / self.R_M
        return I_HP_min_I_source, I_HP_max_I_source
    
    def get_constraints_I_BT2I_HP(self, x_FAN:np.ndarray) -> np.ndarray:
        # TODO implement
        pass
    
    def _input_bounds(self, x:np.ndarray, u:np.ndarray) -> np.ndarray:
        u_bounded = np.copy(u)

        # HP current bounds
        I_HP_min_I_source, I_HP_max_I_source = self.get_constraints_U_BT2I_HP(x[2] - x[1])
        u_bounded[0] = np.clip(u_bounded[0], I_HP_min_I_source, I_HP_max_I_source)
        u_bounded[0] = np.clip(u_bounded[0], -self.I_HP_max, self.I_HP_max)

        # Fan duty cycle bounds
        u_bounded[1] = np.clip(u_bounded[1], 0.0, 1.0)
        return u_bounded
    
    def _states_bounds(self, x:np.ndarray) -> np.ndarray:
        x_bounded = np.copy(x)

        x_bounded[0] = np.clip(x[0], 0.0, 1.0)
        x_bounded[1] = np.clip(x[1], 0.0, conv_temp(100.0, 'C', 'K'))
        x_bounded[2] = np.clip(x[2], 0.0, conv_temp(100.0, 'C', 'K'))
        return x_bounded
    
    def save_linearized_model(self, directory:str, type:str, Ts:float=None) -> None:
        if type == 'continuous':
            variables = self.get_continuous_linearization()
            names = ['A', 'B', 'h', 'C', 'D', 'l']
        elif type == 'discrete':
            if Ts is None:
                raise ValueError("Ts must be provided for discrete linearization")
            variables = self.get_discrete_linearization(Ts)
            names = ['Ad', 'Bd', 'h', 'Cd', 'Dd', 'l']
        else:
            raise ValueError("Type must be either 'continuous' or 'discrete'")

        for name, variable in zip(names, variables):
            np.savetxt(f"{directory}{name}.csv", variable, delimiter=',')

    @property
    def get_initial_state(self) -> np.ndarray:
        return self.x0
    
    @property
    def get_operational_state(self) -> np.ndarray:
        return self.x_op
    
    @property
    def get_operational_input(self) -> np.ndarray:
        return self.u_op