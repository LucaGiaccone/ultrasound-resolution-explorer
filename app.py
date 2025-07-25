import streamlit            as st
import numpy                as np
import plotly.graph_objects as go

from scipy.interpolate import griddata

# Loading measured dataset
X         = np.load("X.npy")         # PARAMETERS: shape (N, 4)
y_axial   = np.load("y_axial.npy")   # shape (N,)
y_lateral = np.load("y_lateral.npy") 

X[:, 1]    /= 1e6 # converting to MHz
param_names = ["Depth (mm)", "Center frequency (MHz)", "Voltage (V)", "Number of angles"]
name_to_idx = {name: i for i, name in enumerate(param_names)}

st.set_page_config(layout="wide")
st.title("Ultrasound Resolution Explorer")

# Sidebar
x_var = st.sidebar.selectbox("X axis variable", param_names, index=0)
y_var = st.sidebar.selectbox("Y axis variable", [p for p in param_names if p != x_var], index=1)

fixed_vars = [p for p in param_names if p not in {x_var, y_var}]

fixed_values = {} # (fixed variable values chosen from actual measured unique values)
for var in fixed_vars:
    idx = name_to_idx[var]
    unique_vals = np.unique(X[:, idx])
    fixed_values[var] = st.sidebar.selectbox(f"Fixed {var}", unique_vals)

# Filter dataset by fixed variables 
mask = np.ones(len(X), dtype=bool)
for var, val in fixed_values.items():
    idx = name_to_idx[var]
    mask &= (X[:, idx] == val)

X_filtered         = X[mask]
y_axial_filtered   = y_axial[mask]
y_lateral_filtered = y_lateral[mask]

if len(X_filtered) < 3:
    st.warning("Not enough data points for interpolation with current selections. Try other fixed values.")
    st.stop()

# Setting plots support
x_idx, y_idx = name_to_idx[x_var], name_to_idx[y_var]
x_vals       = X_filtered[:, x_idx]
y_vals      = X_filtered[:, y_idx]

x_min, x_max = st.sidebar.slider( # plot ranges
    f"{x_var} range", float(x_vals.min()), float(x_vals.max()), (float(x_vals.min()), float(x_vals.max()))
)
y_min, y_max = st.sidebar.slider(
    f"{y_var} range", float(y_vals.min()), float(y_vals.max()), (float(y_vals.min()), float(y_vals.max()))
)

grid_x, grid_y = np.meshgrid(   # Grid for interpolation
    np.linspace(x_min, x_max, 60),
    np.linspace(y_min, y_max, 60),
)

# interp method and tabs
interp_method    = st.sidebar.selectbox("Interpolation method", ["linear", "nearest", "cubic"])
tab1, tab2, tab3 = st.tabs(["Axial resolution", "Lateral resolution", "Resolution estimation"])

def plot_surface(tab, y_data, title, colorscale):
    grid_z = griddata(
        points=np.vstack((x_vals, y_vals)).T,
        values=y_data,
        xi=(grid_x, grid_y),
        method=interp_method,
    )

    in_range_mask = (x_vals >= x_min) & (x_vals <= x_max) & (y_vals >= y_min) & (y_vals <= y_max)

    with tab:
        fig = go.Figure()
    
        fig.add_trace(
            go.Surface(
                x=grid_x,
                y=grid_y,
                z=grid_z,
                colorscale=colorscale,
                opacity=0.8,
                name="Interpolated Surface",
                showscale=True,
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=x_vals[in_range_mask],
                y=y_vals[in_range_mask],
                z=y_data[in_range_mask],
                mode="markers",
                marker=dict(size=4, color="red"),
                name="Measured Points",
            )
        )
        fig.update_layout(
            scene=dict(
                xaxis_title=x_var,
                yaxis_title=y_var,
                zaxis_title=title,
            ),
            margin=dict(l=0, r=0, t=40, b=0),
            height=700,
        )
        st.plotly_chart(fig, use_container_width=True)

plot_surface(tab1, y_axial_filtered, "Axial resolution (mm)", "Viridis")
plot_surface(tab2, y_lateral_filtered, "Lateral resolution (mm)", "Cividis")

with tab3:

    custom_inputs = {}
    for var in param_names:
        idx = name_to_idx[var]
        vals = np.unique(X[:, idx])
        min_val, max_val = vals.min(), vals.max()
        default_val = np.median(vals)

        if var == "Number of angles":

            custom_inputs[var] = st.number_input(
                f"Enter {var}",
                min_value=int(min_val),
                max_value=int(max_val),
                value=int(default_val),
                step=1,
                format="%d"
            )
        else:
      
            custom_inputs[var] = st.number_input(
                f"Enter {var}",
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(default_val),
                step=0.1,
                format="%.2f"
            )

    inp_vec = np.array([custom_inputs[var] for var in param_names])


    k = st.number_input(
        "Number of nearest neighbors to consider",
        min_value=1,
        max_value=min(len(X), 50),
        value=5,
        step=1,
    )

    X_min  = X.min(axis=0)
    X_max  = X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min)

    inp_vec      = np.array([custom_inputs[var] for var in param_names])
    inp_vec_norm = (inp_vec - X_min) / (X_max - X_min)

    dists           = np.linalg.norm(X_norm - inp_vec_norm, axis=1)  # Compute distances in normalized space
    nearest_indices = np.argsort(dists)[:k]
    nearest_dists   = dists[nearest_indices]
    nearest_dists   = np.where(nearest_dists == 0, 1e-8, nearest_dists) 

    weights  = 1 / nearest_dists
    weights /= weights.sum()

    # Weighted NN resolution average 
    axial_estimate   = np.dot(weights, y_axial[nearest_indices])
    lateral_estimate = np.dot(weights, y_lateral[nearest_indices])


    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style='padding: 20px; border-radius: 10px; background-color: #f9f9f9;
                        border-left: 5px solid #4CAF50; text-align: center;'>
                <h3 style='color:#4CAF50; margin-bottom: 10px;'>Axial resolution</h3>
                <p style='font-size: 24px; font-weight: bold;'>{axial_estimate:.5f} mm</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style='padding: 20px; border-radius: 10px; background-color: #f9f9f9;
                        border-left: 5px solid #2196F3; text-align: center;'>
                <h3 style='color:#2196F3; margin-bottom: 10px;'>Lateral resolution</h3>
                <p style='font-size: 24px; font-weight: bold;'>{lateral_estimate:.5f} mm</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.checkbox("Show details of nearest neighbors"): # NN details
        import pandas as pd

        df_summary                       = pd.DataFrame(X[nearest_indices], columns=param_names)
        df_summary["Distance"]           = nearest_dists
        df_summary["Axial Resolution"]   = y_axial[nearest_indices]
        df_summary["Lateral Resolution"] = y_lateral[nearest_indices]
        df_summary["Weight"]             = weights

        st.dataframe(df_summary.round(4))

