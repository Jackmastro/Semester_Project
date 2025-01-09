import numpy as np
import pandas as pd
import sympy as sp
from IPython.display import display, Markdown
from scipy.constants import convert_temperature as conv_temp
from scipy.signal import cont2discrete

from classes.LED_params import LEDparams

from data import load_coefficients
from output_notebooks import save_matrices2csv

class Model:
    def __init__(self, LEDparams:LEDparams, x0:np.ndarray, T_amb0:float=conv_temp(25.0, 'C', 'K')) -> None:
        """
        x0 : Initial state

        u1: I_HP
        u2: x_FAN

        x1: x_SoC
        x2: T_HP_c
        x3: T_HP_h
        """
        # Characterization between cooling or heating for the Peltier module (heat pump)
        self.HP_in_cooling = True

        # Initial condition
        self.x0 = x0
        self.x_prev = x0
        self.x_next = None
        self.u = np.array([0.0,
                           0.0]) # needed for symbolic initialization

        # Parameters initialization
        self._init_params(LEDparams, T_amb0)

        # Dictionary to connect symbolic variables with values
        self.params_values = {
            'n_BT':         self.n_BT, # Battery
            'Q_max':        self.Q_BT_max,
            'R_BT':         self.R_BT,
            'P_rest':       self.P_rest,
            'Q_rest':       self.Q_rest,
            'I_rest':       self.I_rest,
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
            'P_rad':        self.P_rad,
            'S_M':          self.S_M, # HP
            'R_M':          self.R_M,
            'K_M':          self.K_M,
            'cp_Al':        self.cp_Al, # Thermal
            'T_amb':        self.T_amb,
            'm_top':        self.m_2 + self.m_4,
            'm_bot':        self.m_1,
            'R_floor':      self.R_floor,
            'R_top_cell':   self.R_top_cell,
            'R_cell_amb':   self.R_cell_amb,
        }

        # Operational point initialization
        self._init_operational_points()

        # Symbolic initialization
        self._init_sym_model()

    def _init_params(self, LEDparams:LEDparams, T_amb:float) -> None:
        # Battery parameters
        self.P_rest = 1.0 # W TODO get better value
        self.I_rest = 0.5 # A TODO get better value
        self.U_rest = self.P_rest / self.I_rest # V
        self.n_BT = 2
        self.Q_BT_max = 3.0 * 3600 # As (used conversion: Ah = 3600 As)
        self.R_BT = 0.1 # Ohm
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
        self.P_rad = LEDparams.P_rad # W

        # Thermal parameters
        self.T_amb = T_amb # K
        self.cp_Al = 897.0 # J/kgK https://en.wikipedia.org/wiki/6061_aluminium_alloy
        # self.cp_H2O = 4180.0 # J/kgK https://www.engineeringtoolbox.com/specific-heat-capacity-water-d_660.html
        # self.cp_air = 1006.0 # J/kgK https://www.engineeringtoolbox.com/air-specific-heat-capacity-d_705.html?vA=37&degree=C&pressure=1bar#

        # Top thermal parameters - Diffuser
        # TODO estimation of parameters
        self.R_top_cell = 5.0 # K/W
        self.R_cell_amb = 25.0 # K/W

        # Top Al thermal parameters
        self.m_2 = 0.0638 # kg
        self.m_4 = 0.0642 # kg
        self.Q_rest = self.P_rest # W # TODO only valid when battery attached. Case when attached via cable

        # Bottom Al thermal parameters - Heat sink
        self.m_1 = 0.0876 # kg
        self.R_floor = 2.6 # K/W # TODO estimated real time

        # Heat pump - peltier module
        HP_params = load_coefficients('heat_pump\\HP_fitted_coefficients.csv')
        self.R_M = HP_params["R_M"].iloc[0] # Ohm
        self.S_M = HP_params["S_M"].iloc[0] # V/K
        self.K_M = HP_params["K_M"].iloc[0] # W/K

        self.DeltaT_max = HP_params["DeltaT_max"].iloc[0] # K

        I_HP_max_datasheet = HP_params["I_max"].iloc[0] # A
        I_HP_max_electronics = 3.0 # A when attached to the battery
        self.I_HP_max = min(I_HP_max_datasheet, I_HP_max_electronics) # A
        self.I_HP_min = -self.I_HP_max

    def _init_operational_points(self) -> None:
        x_SoC     = 0.85
        T_cold    = conv_temp(self.T_amb, 'C', 'K') # K
        T_hot     = conv_temp(self.T_amb + 20.0, 'C', 'K') # K
        I_HP_cool = 1.2 # A
        x_FAN     = 1.0 # duty cycle

        # COOLING
        self.x_op_cool = np.array([x_SoC,
                                   T_cold,
                                   T_hot])
        self.u_op_cool = np.array([I_HP_cool,
                                   x_FAN])
        
        # HEATING
        self.x_op_heat = np.array([x_SoC,
                                   T_hot,
                                   T_cold])
        self.u_op_heat = np.array([-I_HP_cool,
                                   x_FAN])

    def _init_sym_model(self) -> None:
        # States
        x_SoC, T_top, T_bot = sp.symbols('x_SoC, T_top, T_bot')
        self.sym_x = sp.Matrix([x_SoC,
                                T_top,
                                T_bot])

        # Control inputs
        I_HP, x_FAN = sp.symbols('I_HP, x_FAN')
        self.sym_u  = sp.Matrix([I_HP,
                                 x_FAN])

        # Exogenous input
        T_amb = sp.symbols('T_amb')

        ### Parameters
        ## Peltier module - heat pump
        S_M, R_M, K_M = sp.symbols('S_M, R_M, K_M')

        ## FAN
        U_FAN, I_FAN = sp.symbols('U_FAN, I_FAN')
        q, m         = sp.symbols('q_FAN, m_FAN')
        # a, b, c, d   = sp.symbols('a_FAN, b_FAN, c_FAN, d_FAN')

        ## LED
        I_LED, x_LED, P_rad = sp.symbols('I_LED, x_LED, P_rad')

        ## Battery
        n, Q_max, R_BT, P_rest, I_rest = sp.symbols('n_BT, Q_max, R_BT, P_rest, I_rest')
        a3, a2, a1, a0                 = sp.symbols('a3, a2, a1, a0')
        U_oc = a3 * x_SoC**3 + a2 * x_SoC**2 + a1 * x_SoC + a0 # V
        # display(Markdown(r"$U_{oc}(x_{SoC}):$"), U_oc.subs(self.params_values))

        ## Thermal
        cp_Al = sp.symbols('cp_Al')

        ## Top static
        R_top_cell, R_cell_amb = sp.symbols('R_top_cell, R_cell_amb')

        ## Top dynamic
        m_top = sp.symbols('m_top')

        ## Bottom dynamic
        m_bot, Q_rest = sp.symbols('m_bot, Q_rest')

        ## Bottom static
        R_floor = sp.symbols('R_floor')

        ### Calculations
        ## FAN
        P_FAN = I_FAN * U_FAN * x_FAN # W
        R_FAN_alpha = q + m * x_FAN # K/W

        ## Top static
        Q_LED_cell = (T_top - T_amb) / (R_cell_amb + R_top_cell) # W

        ## Bottom static
        R_eq_bottom = (R_floor * R_FAN_alpha) / (R_floor + R_FAN_alpha) # K/W equivalent parallel resistance
        Q_bot_amb = (T_bot - T_amb) / R_eq_bottom
        # display(Markdown(r"$R_{eq}(x_{FAN}=1):$"), R_eq_bottom.subs(self.params_values).subs({x_FAN: 1.0}))
        # display(Markdown(r"$R_{eq}(x_{FAN}=0):$"), R_eq_bottom.subs(self.params_values).subs({x_FAN: 0.0}))

        ## Output
        T_cell = R_cell_amb * Q_LED_cell + T_amb # K
        # display(Markdown(r"$T_{cell}:$"), T_cell)

        # R_FAN_alpha = c + (1 / (x_FAN + a) + b - c) / (1 + sp.exp(-d * x_FAN)) # K/W
        # display(Markdown(r"$R_{air}^\alpha(x_{FAN}):$"), R_FAN_alpha)
        # display(Markdown(r"$R_{air}^\alpha(x_{FAN}=1):$"), R_FAN_alpha.subs(self.params_values).subs({x_FAN: 1.0}))
        # display(Markdown(r"$R_{air}^\alpha(x_{FAN}=0):$"), R_FAN_alpha.subs(self.params_values).subs({x_FAN: 0.0}))

        ## Battery
        # I_BT = (U_oc + R_BT*I_LED*x_LED - sp.sqrt(U_oc**2 - 2*R_BT*I_LED*x_LED*U_oc + (R_BT*I_LED*x_LED)**2 - 4*R_BT*(P_rest + P_FAN + sp.sqrt(P_HP**2)))) / (2*R_BT) # A
        I_BT = (I_LED*x_LED + I_FAN*x_FAN + sp.sqrt(I_HP**2) + I_rest) / n # A
        U_BT = U_oc - R_BT * I_BT # V
        P_BT = U_BT * I_BT # W
        # display(Markdown(r"$U_{BT}:$"), U_BT.subs(self.params_values))
        # display(Markdown(r"$U_{BT}^{}:$"), U_BT.subs(self.params_values).subs({I_HP:3.0,  x_FAN: 1.0, x_SoC: 1.0}))
        # display(Markdown(r"$U_{BT}^{max}:$"), U_BT.subs(self.params_values).subs({I_HP:3.0,  x_FAN: 0.0, x_SoC: 1.0}))
        # display(Markdown(r"$U_{BT}^{min}:$"), U_BT.subs(self.params_values).subs({I_HP:3.0,  x_FAN: 0.0, x_SoC: 0.0}))
        # display(Markdown(r"$U_{BT}^{}:$"), U_BT.subs(self.params_values).subs({I_HP:-3.0, x_FAN: 1.0, x_SoC: 1.0}))
        # display(Markdown(r"$U_{BT}^{max}:$"), U_BT.subs(self.params_values).subs({I_HP:-3.0, x_FAN: 0.0, x_SoC: 1.0}))
        # display(Markdown(r"$U_{BT}^{min}:$"), U_BT.subs(self.params_values).subs({I_HP:-3.0, x_FAN: 0.0, x_SoC: 0.0}))

        ## LED
        P_LED = I_LED * U_BT * x_LED # W
        Q_LED = P_LED - P_rad # W

        ############################### INDEPENDENT of cooling or heating
        ### Output
        # Symbolic
        self.g_symb = sp.Matrix([T_top])
        # display(Markdown(r"$\dot{x} = f(x, u):$"), self.f_symb)
        # display(Markdown(r"$\dot{x} = f(x, u):$"), sp.latex(self.f_symb))
        # display(Markdown(r"$y = g(x, u):$"), self.g_symb)

        # Linearization
        self.C_symb = self.g_symb.jacobian(self.sym_x)
        self.D_symb = self.g_symb.jacobian(self.sym_u)
        # display(Markdown(r"$A = \nabla_x f:$"), self.A_symb.subs(self.params_values).subs({x_SoC: 0.85, T_top: 25.0, T_bot: 55.0, x_FAN: 1.0}))
        # display(Markdown(r"$B = \nabla_u f:$"), self.B_symb.subs(self.params_values).subs({x_SoC: 0.85, T_top: 25.0, T_bot: 55.0, x_FAN: 1.0}))

        # Numerical functions with parameters already inserted
        self.g_num = sp.lambdify((self.sym_x, self.sym_u), self.g_symb.subs(self.params_values), modules="numpy")

        self.C_num = sp.lambdify((self.sym_x, self.sym_u), self.C_symb.subs(self.params_values), modules="numpy")
        self.D_num = sp.lambdify((self.sym_x, self.sym_u), self.D_symb.subs(self.params_values), modules="numpy")

        # Numerical functions for plots - values
        self.U_BT_num   = sp.lambdify((self.sym_x, self.sym_u), U_BT.subs(self.params_values), modules="numpy")
        self.U_oc_num   = sp.lambdify((self.sym_x, self.sym_u), U_oc.subs(self.params_values), modules="numpy")
        self.I_BT_num   = sp.lambdify((self.sym_x, self.sym_u), I_BT.subs(self.params_values), modules="numpy")
        self.T_cell_num = sp.lambdify((self.sym_x, self.sym_u), T_cell.subs(self.params_values), modules="numpy")

        ## Peltier module - heat pump: split the model in two parts, depending whether we are cooling or heating the top part
        ############################### COOLING of T_top
        Delta_T_HP_cool = T_bot - T_top
        U_HP_cool       = S_M*Delta_T_HP_cool + R_M*I_HP # V
        P_HP_cool       = U_HP_cool*I_HP # W
        Q_top_cool      = S_M*T_top*I_HP - 0.5*R_M*I_HP**2 - K_M*Delta_T_HP_cool # W
        Q_bot_cool      = Q_top_cool + P_HP_cool # W
        COP_cool        = sp.sqrt(Q_top_cool**2) / sp.sqrt(P_HP_cool**2) # Condition for P_HP close to zero in get_values

        ### Nonlinear ODEs
        dx_SoC_dt_cool = - I_BT / (n * Q_max)
        dT_top_dt_cool = (1 / (m_top * cp_Al)) * (Q_LED - Q_LED_cell - Q_top_cool)
        dT_bot_dt_cool = (1 / (m_bot * cp_Al)) * (Q_bot_cool - Q_bot_amb + Q_rest)

        ### Dynamics
        # Symbolic
        self.f_symb_cool = sp.Matrix([dx_SoC_dt_cool,
                                      dT_top_dt_cool,
                                      dT_bot_dt_cool])

        # Linearization
        self.A_symb_cool = self.f_symb_cool.jacobian(self.sym_x)
        self.B_symb_cool = self.f_symb_cool.jacobian(self.sym_u)

        # Numerical functions with parameters already inserted
        self.f_num_LED_off = sp.lambdify((self.sym_x, self.sym_u), self.f_symb_cool.subs({'x_LED': 0.0}).subs(self.params_values), modules="numpy") # needed for extending the plots to negative times
        self.f_num_cool    = sp.lambdify((self.sym_x, self.sym_u), self.f_symb_cool.subs(self.params_values), modules="numpy")

        self.A_num_cool    = sp.lambdify((self.sym_x, self.sym_u), self.A_symb_cool.subs(self.params_values), modules="numpy")
        self.B_num_cool    = sp.lambdify((self.sym_x, self.sym_u), self.B_symb_cool.subs(self.params_values), modules="numpy")

        # Numerical functions for plots - values
        self.U_HP_num_cool  = sp.lambdify((self.sym_x, self.sym_u), U_HP_cool.subs(self.params_values), modules="numpy")
        self.Q_top_num_cool = sp.lambdify((self.sym_x, self.sym_u), Q_top_cool.subs(self.params_values), modules="numpy")
        self.COP_num_cool   = sp.lambdify((self.sym_x, self.sym_u), COP_cool.subs(self.params_values), modules="numpy")

        # Needed to check condition on COP
        self.P_HP_num_cool = sp.lambdify((self.sym_x, self.sym_u), P_HP_cool.subs(self.params_values), modules="numpy")

        ############################### HEATING of T_top
        Delta_T_HP_heat = T_top - T_bot
        U_HP_heat       = S_M*Delta_T_HP_heat + R_M*I_HP # V
        P_HP_heat       = U_HP_heat*I_HP # W
        Q_top_heat      = -S_M*T_top*I_HP - 0.5*R_M*I_HP**2 + K_M*Delta_T_HP_heat # W
        Q_bot_heat      = Q_top_heat + P_HP_heat # W
        COP_heat        = sp.sqrt(Q_top_heat**2) / sp.sqrt(P_HP_heat**2) # Condition for P_HP close to zero in get_values

        ### Nonlinear ODEs
        dx_SoC_dt_heat = - I_BT / (n * Q_max)
        dT_top_dt_heat = (1 / (m_top * cp_Al)) * (Q_LED - Q_LED_cell - Q_top_heat)
        dT_bot_dt_heat = (1 / (m_bot * cp_Al)) * (Q_bot_heat - Q_bot_amb + Q_rest)
        
        ### Dynamics
        # Symbolic
        self.f_symb_heat = sp.Matrix([dx_SoC_dt_heat,
                                        dT_top_dt_heat,
                                        dT_bot_dt_heat])

        # Linearization
        self.A_symb_heat = self.f_symb_heat.jacobian(self.sym_x)
        self.B_symb_heat = self.f_symb_heat.jacobian(self.sym_u)

        # Numerical functions with parameters already inserted
        self.f_num_heat    = sp.lambdify((self.sym_x, self.sym_u), self.f_symb_heat.subs(self.params_values), modules="numpy")

        self.A_num_heat    = sp.lambdify((self.sym_x, self.sym_u), self.A_symb_heat.subs(self.params_values), modules="numpy")
        self.B_num_heat    = sp.lambdify((self.sym_x, self.sym_u), self.B_symb_heat.subs(self.params_values), modules="numpy")

        # Numerical functions for plots - values
        self.U_HP_num_heat  = sp.lambdify((self.sym_x, self.sym_u), U_HP_heat.subs(self.params_values), modules="numpy")
        self.Q_top_num_heat = sp.lambdify((self.sym_x, self.sym_u), Q_top_heat.subs(self.params_values), modules="numpy")
        self.COP_num_heat   = sp.lambdify((self.sym_x, self.sym_u), COP_heat.subs(self.params_values), modules="numpy")

        # Needed to check condition on COP
        self.P_HP_num_heat = sp.lambdify((self.sym_x, self.sym_u), P_HP_heat.subs(self.params_values), modules="numpy")

    def get_continuous_linearization(self, T_ref:float, T_amb:float, xss:np.ndarray=None, uss:np.ndarray=None) -> np.ndarray:
        self.T_cell_ref = T_ref
        
        # COOLING
        # if T_ref <= T_amb:
        #     if xss is None:
        #         xss = self.x_op_cool
        #     if uss is None:
        #         uss = self.u_op_cool

        #     A = self.A_num_cool(xss, uss)
        #     B = self.B_num_cool(xss, uss)
        #     h = self.f_num_cool(xss, uss).reshape(-1,) - A @ xss - B @ uss

        #  # HEATING    
        # else:
        #     if xss is None:
        #         xss = self.x_op_heat
        #     if uss is None:
        #         uss = self.u_op_heat

        #     A = self.A_num_heat(xss, uss)
        #     B = self.B_num_heat(xss, uss)
        #     h = self.f_num_heat(xss, uss).reshape(-1,) - A @ xss - B @ uss
        if xss is None:
            xss = self.x_op_cool
        if uss is None:
            uss = self.u_op_cool

        A = self.A_num_cool(xss, uss)
        B = self.B_num_cool(xss, uss)
        h = self.f_num_cool(xss, uss).reshape(-1,) - A @ xss - B @ uss

        C = self.C_num(xss, uss)
        D = self.D_num(xss, uss)
        l = self.g_num(xss, uss).reshape(-1,) - C @ xss - D @ uss

        return np.array(A).astype(np.float32), np.array(B).astype(np.float32), np.array(h).astype(np.float32), np.array(C).astype(np.float32), np.array(D).astype(np.float32), np.array(l).astype(np.float32)
    
    def get_discrete_linearization(self, T_ref:float, T_amb:float, dt_d:float, xss:np.ndarray=None, uss:np.ndarray=None) -> np.ndarray:
        if xss is None:
            xss = self.x_op_cool
        if uss is None:
            uss = self.u_op_cool # TODO do only once instead of twice

        A, B, _, C, D, _ = self.get_continuous_linearization(T_ref, T_amb, xss, uss)
        A_d, B_d, C_d, D_d, _ = cont2discrete((A, B, C, D), dt=dt_d, method='zoh')

        # COOLING
        if T_ref <= T_amb:
            h_d = self.f_num_cool(xss, uss).reshape(-1,) - A_d @ xss - B_d @ uss

         # HEATING    
        else:
            h_d = self.f_num_heat(xss, uss).reshape(-1,) - A_d @ xss - B_d @ uss

        l_d = self.g_num(xss, uss).reshape(-1,) - C_d @ xss - D_d @ uss
        
        return np.array(A_d).astype(np.float32), np.array(B_d).astype(np.float32), np.array(h_d).astype(np.float32), np.array(C_d).astype(np.float32), np.array(D_d).astype(np.float32), np.array(l_d).astype(np.float32)
    
    def _update_HP_op_space(self, x:np.ndarray, u:np.ndarray) -> None:
        # COOLING
        if self.Q_top_num_cool(x, u) >= 0:
            self.HP_in_cooling = True

        # HEATING
        else:
            self.HP_in_cooling = True
    
    def _dynamics_f(self, x:np.ndarray, u:np.ndarray, LED_off:bool=False) -> np.ndarray:
        if LED_off:
            return np.array(self.f_num_LED_off(x, u)).flatten()
        else:
            # COOLING
            if self.HP_in_cooling:
                return np.array(self.f_num_cool(x, u)).flatten()
            
            # HEATING
            else:
                return np.array(self.f_num_heat(x, u)).flatten()

    def _observer_g(self, x:np.ndarray, u:np.ndarray) -> np.ndarray:
        return np.array(self.g_num(x, u)).flatten()

    def discretized_update(self, u:np.ndarray, dt:float, LED_off:bool=False) -> np.ndarray:
        """
        Update the states using Runge-Kutta of 4th order integration.
        """
        x = self.x_prev

        # Bound input
        u_bounded = self._input_bounds(x, u)

        # Check whether HP in cooling or heating
        self._update_HP_op_space(x, u_bounded)

        # Runge-Kutta 4th order
        k = np.zeros((4, len(x)))
        k[0] = self._dynamics_f(x, u_bounded, LED_off=LED_off)
        k[1] = self._dynamics_f(x + 0.5 * dt * k[0], u_bounded, LED_off=LED_off)
        k[2] = self._dynamics_f(x + 0.5 * dt * k[1], u_bounded, LED_off=LED_off)
        k[3] = self._dynamics_f(x + dt * k[2], u_bounded, LED_off=LED_off)

        self.x_next = x + (dt / 6.0) * (k[0] + 2 * k[1] + 2 * k[2] + k[3])

        # Bound states
        self.x_next = self._states_bounds(self.x_next)

        # Update
        self.u = u_bounded
        self.x_prev = self.x_next
        
        return self.x_next, self.u
    
    def get_output(self) -> np.ndarray:
        return self._observer_g(self.x_prev, self.u)
    
    def get_values(self, x:np.ndarray, u:np.ndarray, LED_off:bool=False) -> dict:
        # To avoid multiple calls of _update_HP_op_space when running the simulation, here no update of HP_in_cooling

        # Negative times
        if LED_off:
            x_LED = 0.0
            COP = np.nan

            # COOLING
            if self.HP_in_cooling:
                U_HP = self.U_HP_num_cool(x, u)
            
            # HEATING
            else:
                U_HP = self.U_HP_num_heat(x, u)
        
        # Simulation
        else:
            x_LED = self.x_LED_tot

            tol = 1e-5

            # COOLING
            if self.HP_in_cooling:
                U_HP = self.U_HP_num_cool(x, u)

                # Condition for COP with P_HP close to zero
                if abs(self.P_HP_num_cool(x, u)) < tol:
                    COP = np.nan
                else:
                    COP = self.COP_num_cool(x, u)
            
            # HEATING
            else:
                U_HP = self.U_HP_num_heat(x, u)

                # Condition for COP with P_HP close to zero
                if abs(self.P_HP_num_heat(x, u)) < tol:
                    COP = np.nan
                else:
                    COP = self.COP_num_heat(x, u)

        return {
            'U_BT':        self.U_BT_num(x, u),
            'U_oc':        self.U_oc_num(x, u),
            'U_HP':        U_HP,
            'COP':         COP,
            'I_BT':        self.I_BT_num(x, u),
            'T_cell':      self.T_cell_num(x, u),
            'x_LED':       x_LED,
        }
    
    def get_constraints_U_BT2I_HP(self, delta_T:np.ndarray) -> np.ndarray:
        # Zero order approximation for U_BT
        U_BT = self.U_BT_num(self.x_op_cool, self.u_op_cool) # TODO use max and min values USING COOL IS FINE?
        I_HP_max_U_BT = (U_BT - self.S_M * delta_T) / self.R_M
        I_HP_min_U_BT = (- U_BT - self.S_M * delta_T) / self.R_M
        return I_HP_min_U_BT, I_HP_max_U_BT
    
    def get_constraints_I_BT2I_HP(self, u_bounded:np.ndarray) -> np.ndarray:
        # Assuming that the current of the fan is satisfied
        C = self.I_LED * self.x_LED_tot + self.I_FAN * u_bounded[1] + self.I_rest

        if u_bounded[0] >= 0:
            I_HP_min_I_BT = self.I_BT_min - C
            I_HP_max_I_BT = self.I_BT_max - C
        else:
            I_HP_min_I_BT = C - self.I_BT_max
            I_HP_max_I_BT = C - self.I_BT_min

        return I_HP_min_I_BT, I_HP_max_I_BT
    
    def _input_bounds(self, x:np.ndarray, u:np.ndarray) -> np.ndarray:
        u_bounded = np.copy(u)

        # Fan duty cycle bounds
        u_bounded[1] = np.clip(u_bounded[1], 0.0, 1.0)

        # HP current bounds
        I_HP_min_U_BT, I_HP_max_U_BT = self.get_constraints_U_BT2I_HP(x[2] - x[1])
        u_bounded[0] = np.clip(u_bounded[0], I_HP_min_U_BT, I_HP_max_U_BT)
        u_bounded[0] = np.clip(u_bounded[0], self.I_HP_min, self.I_HP_max)

        I_HP_min_I_BT, I_HP_max_I_BT = self.get_constraints_I_BT2I_HP(u_bounded)
        u_bounded[0] = np.clip(u_bounded[0], I_HP_min_I_BT, I_HP_max_I_BT)

        return u_bounded
    
    def _states_bounds(self, x:np.ndarray) -> np.ndarray:
        x_bounded = np.copy(x)

        x_bounded[0] = np.clip(x[0], 0.0, 1.0)
        x_bounded[1] = np.clip(x[1], 0.0, conv_temp(100.0, 'C', 'K'))
        x_bounded[2] = np.clip(x[2], 0.0, conv_temp(100.0, 'C', 'K'))
        return x_bounded
    
    def save_linearized_model(self, type:str, T_ref:float=None, T_amb:float=None, Ts:float=None) -> None:
        if T_ref is None:
            T_ref = self.T_cell_ref # TODO could be problematic if not given in __init__
        if T_amb is None:
            T_amb = self.T_amb

        # Conditions on continous or discrete
        if type == 'continuous':
            variables = self.get_continuous_linearization(T_ref, T_amb)
            names = ['A', 'B', 'h', 'C', 'D', 'l']

        elif type == 'discrete':
            if Ts is None:
                raise ValueError("Ts must be provided for discrete linearization")
            variables = self.get_discrete_linearization(T_ref, T_amb, Ts)
            names = ['Ad', 'Bd', 'hd', 'Cd', 'Dd', 'ld']

        else:
            raise ValueError("Type must be either 'continuous' or 'discrete'")

        # Save matrices
        save_matrices2csv(names, variables)

    @property
    def get_initial_state(self) -> np.ndarray:
        return self.x0
    
    @property
    def get_operational_state(self) -> np.ndarray:
        return self.x_op_cool, self.x_op_heat
    
    @property
    def get_operational_input(self) -> np.ndarray:
        return self.u_op_cool, self.u_op_heat