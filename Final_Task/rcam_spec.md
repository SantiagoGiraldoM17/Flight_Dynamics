# RCAM 6-DOF Model Spec

## Constants
g=9.81; rho=1.225; P0=101325; T0=288.15
S=260; St=64; lt=24.8; l=6.6; b=44.8; cbar=6.6
XAPT1=0.0; YAPT1=7.94; ZAPT1=1.9
XAPT2=0.0; YAPT2=-7.94; ZAPT2=1.9
m_nom=120000; alpha0=11.5*pi/180; deps_dalpha=0.25

## State x[12] (SI, rad, m, m/s)
x[0..2]=p,q,r            # body rates
x[3..5]=phi,theta,psi    # Euler ZYX (yaw,pitch,roll order applied)
x[6..8]=uB,vB,wB         # body velocity
x[9..11]=X,Y,Z           # earth pos; Z+down; h=-Z

## Input u[8]
u[0..2]=dA,dE,dR         # rad
u[3..4]=dTH1,dTH2        # 0..1
u[5..7]=WxE,WyE,WzE      # earth-frame wind m/s

## Inertia (kg·m²)
I = m * [[40.07,0,2.098],[0,64,0],[2.098,0,99.92]]

## Rotation R_BV (vehicle-carried -> body), c=cos s=sin
R_BV =
[[ cψcθ,            sψcθ,            -sθ   ],
 [ cψsθsφ-sψcφ,     sψsθsφ+cψcφ,     cθsφ  ],
 [ cψsθcφ+sψsφ,     sψsθcφ-cψsφ,     cθcφ  ]]
# body->earth: R_BV.T

## Airspeed
W_B = R_BV @ [WxE,WyE,WzE]
[ua,va,wa] = [uB,vB,wB] - W_B
V = sqrt(ua²+va²+wa²)
alpha = atan2(wa,ua)
beta  = asin(va/V)

## Aero coefficients
eps     = deps_dalpha*(alpha-alpha0)
alpha_t = alpha - eps + dE + 1.3*q*lt/V
CL_wb   = 5.5*(alpha-alpha0)
CL_t    = (St/S)*3.1*alpha_t
CL      = CL_wb + CL_t
CD      = 0.13 + 0.07*(CL_wb-0.45)²
CY      = -1.6*beta + 0.24*dR

Cl = -1.4*beta
   + (l/V)*(-11*p + 0*q + 5*r)
   + (-0.6*dA + 0*dE + 0.22*dR)

Cm = -0.59 - 3.1*(St*lt/(S*l))*(alpha-eps)
   + (l/V)*(0*p - 4.03*(St*lt²/(S*l²))*q + 0*r)
   + (0*dA - 3.1*(St*lt/(S*l))*dE + 0*dR)

Cn = (1 - alpha*180/(15*pi))*beta
   + (l/V)*(1.7*p + 0*q - 11.5*r)
   + (0*dA + 0*dE - 0.63*dR)

## Aero forces (body axes)
qbar = 0.5*rho*V²
D = CD*qbar*S; Y = CY*qbar*S; L = CL*qbar*S
FxA = L*sα - D*cα*cβ - Y*cα*sβ
FyA =      - D*sβ    + Y*cβ
FzA = -L*cα - D*sα*cβ - Y*sα*sβ

## Aero moments (body axes)
LA = Cl*qbar*S*b
MA = Cm*qbar*S*cbar
NA = Cn*qbar*S*b
M_aero = [LA, MA, NA]
# Optional CoG offset (Δx,Δy,Δz): add cross([Δx,Δy,Δz],[FxA,FyA,FzA]) to M_aero

## Engines
F1 = dTH1*m*g; F2 = dTH2*m*g
F_eng = [F1+F2, 0, 0]
M_eng = cross([XAPT1,YAPT1,ZAPT1],[F1,0,0])
      + cross([XAPT2,YAPT2,ZAPT2],[F2,0,0])

## Gravity (body axes)
F_grav = m*g*[-sθ, cθ*sφ, cθ*cφ]

## Total
F = [FxA,FyA,FzA] + F_eng + F_grav
M = M_aero + M_eng

## EoM
# translational
duB = Fx/m - (q*wB - r*vB)
dvB = Fy/m - (r*uB - p*wB)
dwB = Fz/m - (p*vB - q*uB)
# rotational
omega = [p,q,r]
omega_dot = solve(I, M - cross(omega, I@omega))
dp,dq,dr = omega_dot
# Euler kinematics
dphi   = p + sφ*tθ*q + cφ*tθ*r
dtheta =     cφ*q   - sφ*r
dpsi   =     sφ/cθ*q + cφ/cθ*r
# Position
[dX,dY,dZ] = R_BV.T @ [uB,vB,wB]

## Actuators (first-order lag + saturation)
dA:   tau=0.15, sat=[-25,25] deg
dE:   tau=0.15, sat=[-25,10] deg
dR:   tau=0.30, sat=[-30,30] deg
dTH:  tau=1.5,  sat=[0.5,1.0], rate=±1.6/s
engine_failure: dTH -> 0.5 via 1/(1+3.3s)

## Outputs
h    = -Z
nx   =  Fx/(m*g)
ny   =  Fy/(m*g)
nz   = -Fz/(m*g)
[uV,vV,wV] = R_BV.T @ [uB,vB,wB]
gamma = asin(-wV/V)
chi   = atan2(-vV, uV)
VCAS  ≈ V (no compressibility)

## Uncertainty bounds (for robustness sweeps)
m  ∈ [100000, 150000] kg
Δx ∈ [0.2, 1.25] m
Δy ∈ [-0.2, 0.2] m
Δz ∈ [0, 1.38] m
time_delay ∈ [0.05, 0.2] s

## Safety/envelope limits
alpha_stall = 18 deg
V_min = 1.05*V_stall;  V_stall(m=120000) = 51.1 m/s
phi_max = 30 deg

## Trim init (level flight example)
V0=80; gamma0=0; h0=1000; X0=Y0=0
mass0=120000; (Δx,Δy,Δz)=(0.3,0,0)
alpha0_guess=beta0=phi0=0; theta0=alpha0_guess+gamma0
# solve for dE, dTH such that derivatives = 0 at this condition

## Dryden turbulence (optional, h in meters)
W20 = 15.4   # moderate, m/s
sigma_z = 0.1*W20
if 3 < h < 305:
    sigma_x = sigma_y = sigma_z/(0.177+0.0027*h)**0.4
    Lz = h
    Lx = Ly = h/(0.177+0.0027*h)**1.2
else:  # h >= 305
    sigma_x = sigma_y = sigma_z
    Lx = Ly = Lz = 305
# Spectra (drive shaping filters with white noise):
Phi_x(w) = sigma_x²*(2Lx/pi)*1/(1+(Lx*w)²)
Phi_y(w) = sigma_y²*(Ly/pi)*(1+3(Ly*w)²)/(1+(Ly*w)²)²
Phi_z(w) = sigma_z²*(Lz/pi)*(1+3(Lz*w)²)/(1+(Lz*w)²)²

## Notes
- All angles rad internally; convert deg only at I/O.
- Z positive down (NED-like); altitude h = -Z.
- Throttle dimensionless; thrust = dTH*m*g along +xB.
- alpha in Cn nonlinear term uses alpha in rad; the (alpha*180/(15*pi)) = alpha_deg/15.
- Mean aero chord cbar == generalised length l = 6.6 m.
- Singularity at theta=±pi/2 in Euler kinematics (irrelevant for landing).
