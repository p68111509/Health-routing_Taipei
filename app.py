import streamlit as st
import folium
import pickle
import requests
import networkx as nx
import pandas as pd
from streamlit_folium import st_folium
from shapely.geometry import LineString
from scipy.spatial import KDTree
from branca.element import MacroElement
from jinja2 import Template
from pyproj import Transformer
import base64

# ========== 系統參數 ==========
map_center = [25.04, 121.56]  # 台北市中心
# map_center = [24.1477, 120.6736]  # 台中市中心

# ========== 關閉雙擊放大 ==========
class DisableDoubleClickZoom(MacroElement):
    def __init__(self):
        super().__init__()
        self._template = Template("""
            {% macro script(this, kwargs) %}
                {{this._parent.get_name()}}.doubleClickZoom.disable();
            {% endmacro %}
        """)

# ========== 讀取圖 ==========
@st.cache_resource
def load_graph():
    pkl_path = r"data/雙北基隆路網_濃度與暴露_最大連通版.pkl"
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)

    transformer = Transformer.from_crs("epsg:3826", "epsg:4326", always_xy=True)
    mapping = {}
    for node in list(G.nodes):
        lon, lat = transformer.transform(node[0], node[1])
        mapping[(lat, lon)] = node
        G.nodes[node]["latlon"] = (lat, lon)

    G.graph['latlon_nodes'] = list(mapping.keys())
    G.graph['node_lookup'] = mapping
    return G

# ====== Google Geocoding ======
def geocode(address):
    api_key = "AIzaSyDnbTu8PgUkue5A9uO5aJa3lHZuNUwj6z0"
    full_address = "台灣 " + address
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": full_address, "language": "zh-TW", "key": api_key}
    try:
        response = requests.get(url, params=params).json()
        if response["status"] == "OK":
            location = response["results"][0]["geometry"]["location"]
            return location["lat"], location["lng"]
        else:
            st.warning(f"⚠️ Google 回應：{response['status']} - {response.get('error_message', '無錯誤訊息')}")
            return None
    except Exception as e:
        st.error(f"地址查詢失敗: {e}")
        return None


# ====== Reverse Geocoding（從座標查地址）======
def reverse_geocode(lat, lon):
    api_key = "AIzaSyDnbTu8PgUkue5A9uO5aJa3lHZuNUwj6z0"
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"latlng": f"{lat},{lon}", "language": "zh-TW", "key": api_key}
    try:
        response = requests.get(url, params=params).json()
        if response["status"] == "OK":
            return response["results"][0]["formatted_address"]
        else:
            return ""
    except Exception as e:
        return ""


# ========== 找最近節點 ==========
def find_nearest_node(G, lat, lon, max_dist=0.01):
    kdtree = KDTree(G.graph['latlon_nodes'])
    dist, idx = kdtree.query((lat, lon))
    if dist > max_dist:
        return None
    latlon = G.graph['latlon_nodes'][idx]
    return G.graph['node_lookup'][latlon]

# ========== 路徑計算 ==========
def compute_path(G, start_node, end_node, weight):
    try:
        if isinstance(weight, dict):
            def cost(u, v, d):
                attrs = d.get("attr_dict", {})
                return sum(attrs.get(k, 0) * w for k, w in weight.items())
            path = nx.shortest_path(G, start_node, end_node, weight=cost)
        else:
            path = nx.shortest_path(
                G, start_node, end_node,
                weight=lambda u, v, d: max(0, d.get("attr_dict", {}).get(weight, 0))
            )
    except nx.NetworkXNoPath:
        return None, 0, 0, 0, 0

    total = 0
    PM25_acc = 0
    NO2_acc = 0
    WBGT_acc = 0
    for u, v in zip(path[:-1], path[1:]):
        edge_data = G.get_edge_data(u, v)
        if edge_data and "attr_dict" in edge_data:
            attrs = edge_data["attr_dict"]
            total += attrs.get("length", 0)
            PM25_acc += attrs.get("PM25_expo", 0)
            NO2_acc += attrs.get("NO2_expo", 0)
            WBGT_acc += attrs.get("WBGT_expo", 0)
        else:
            for d in edge_data.values():
                attrs = d.get("attr_dict", {})
                total += attrs.get("length", 0)
                PM25_acc += attrs.get("PM25_expo", 0)
                NO2_acc += attrs.get("NO2_expo", 0)
                WBGT_acc += attrs.get("WBGT_expo", 0)

    return path, total, PM25_acc, NO2_acc, WBGT_acc


# pm25_weight no2_weight WBGT_weight
# "PM25_expo" "NO2_expo" "WBGT_expo"

################################## Streamlit 介面 ##################################
st.set_page_config(layout="wide")

# 初始化狀態（放這裡最安全）
if "disable_inputs" not in st.session_state:
    st.session_state.disable_inputs = False
if "has_routed" not in st.session_state:
    st.session_state.has_routed = False
if "show_pm25_layer" not in st.session_state:
    st.session_state.show_pm25_layer = False

# ==== 自訂按鈕樣式（可選）====
st.markdown("""
    <style>
    button[kind="primary"] {
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 0.4em 1em !important;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns([6, 0.5, 6, 0.5])

with col1:
    st.markdown("""
        <h1 style="
            font-family: 'Noto Sans TC', 'PingFang TC', 'Microsoft JhengHei', sans-serif;
            font-size: 32px;
            font-weight: 800;
            letter-spacing: 1.5px;
            color: black;
            text-align: center;
            margin-bottom: 0px;
            line-height: 1.2;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        ">
            Geo-AI 舒適路徑系統<br>
            <span style="
                font-size: 16px;
                font-weight: 500;
                color: #666666;
            ">
                大台北地區
            </span>
        </h1>
    """, unsafe_allow_html=True)
    # with col_gemini:
    #     # if "set_start_address" in st.session_state:
    #     #     st.session_state.start_address = st.session_state.pop("set_start_address")
    #     gemini_sentense = st.text_input(label="", placeholder="跟 Gemini 說點什麼", key="Gemini")



    if "transport_mode" not in st.session_state:
        st.session_state.transport_mode = "機車"

    G = load_graph()
    if "points" not in st.session_state: st.session_state.points = []
    if "nodes" not in st.session_state: st.session_state.nodes = []

    # 設定CSS來改變底色
    st.markdown("""
        <style>
            .start-address input {
                background-color: #d03c29;
            }
        </style>
    """, unsafe_allow_html=True)

    # 地址輸入框
    row1 = st.columns([1, 1])
    with row1[0]:
        if "set_start_address" in st.session_state:
            st.session_state.start_address = st.session_state.pop("set_start_address")
        start_address = st.text_input(label="", placeholder="🟢起點地址", key="start_address")
    with row1[1]:
        if "set_end_address" in st.session_state:
            st.session_state.end_address = st.session_state.pop("set_end_address")
        end_address = st.text_input(label="", placeholder="🔴終點地址", key="end_address")

    ###### 權重調整
    row4 = st.columns([1,1,1])
    with row4[0]:
        pm25_weight = st.slider("PM₂․₅ 權重 (%)", 0, 100, 50, step=10, key="pm25_weight")
    with row4[1]:
        no2_weight = st.slider("NO₂ 權重 (%)", 0, 100, 30, step=10, key="no2_weight")
    with row4[2]:
        WBGT_weight = st.slider("氣溫 權重 (%)", 0, 100, 80, step=10, key="WBGT_weight")



    ##### 交通方式按鈕 路徑解算 清空選擇
    row2 = st.columns([1, 1, 1, 1])
    with row2[0]:
        st.markdown("""
        <style>
        .select-label-box {
            font-size: 15px;
            font-weight: 600;
            font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif;
            color: #333333;
            background-color: #eeeeee;
            border-radius: 10px;
            padding: 10px 16px;
            text-align: center;
            width: 100%;
            margin-top: 14px;
        }
        </style>
        <div class="select-label-box">交通方式</div>
        """, unsafe_allow_html=True)

    with row2[1]:
        st.markdown("""
        <style>
        div[data-baseweb="select"] > div {
            font-size: 16px !important;
            font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif !important;
        }
        </style>
        """, unsafe_allow_html=True)

        selected_mode = st.selectbox(
            label="交通方式",
            options=["機車", "單車", "步行"],
            index=["機車", "單車", "步行"].index(st.session_state.get("transport_mode", "機車")),
            label_visibility="collapsed",
        )
        st.session_state.transport_mode = selected_mode




    st.markdown("""
        <style>
        div[data-testid="stForm"] {
            padding: 0 !important;
            background-color: transparent !important;
            box-shadow: none !important;
            border: none !important;
        }
        button.pm25-toggle {
            border: 2px solid #cccccc;
            border-radius: 8px;
            padding: 6px 14px;
            font-size: 16px;
            color: black;
            background-color: white;
        }
        button.pm25-toggle.active {
            border-color: red !important;
        }
        </style>
    """, unsafe_allow_html=True)

    with row2[2]: # 🧭 路徑解算
        disabled = st.session_state.disable_inputs  # 按鈕是否鎖定
        st.markdown("""
            <style>
            /* 根據按鈕文字選取目標按鈕 */
            button:has(> div:contains("🧭 路徑解算")) {
                margin-top: 20px;
            }
            </style>
        """, unsafe_allow_html=True)
        if st.button("🧭 路徑解算", disabled=st.session_state.disable_inputs):
            if not start_address.strip():
                st.warning("⚠️ 請輸入起點地址")
            elif not end_address.strip():
                st.warning("⚠️ 請輸入終點地址")
            else:
                # 起點處理
                start_result = geocode(start_address)
                if not start_result:
                    st.warning("⚠️ 起點地址查詢失敗")
                else:
                    start_lat, start_lon = start_result
                    start_node = find_nearest_node(G, start_lat, start_lon)
                    if not start_node:
                        st.warning("⚠️ 起點離路網太遠")
                    else:
                        # 終點處理
                        end_result = geocode(end_address)
                        if not end_result:
                            st.warning("⚠️ 終點地址查詢失敗")
                        else:
                            end_lat, end_lon = end_result
                            end_node = find_nearest_node(G, end_lat, end_lon)
                            if not end_node:
                                st.warning("⚠️ 終點離路網太遠")
                            else:
                                # 一切成功，儲存節點與位置
                                st.session_state.points = [
                                    list(G.nodes[start_node]["latlon"]),
                                    list(G.nodes[end_node]["latlon"]),
                                ]
                                st.session_state.nodes = [start_node, end_node]
                                st.session_state.has_routed = True
                                # 鎖定所有輸入
                                st.session_state.disable_inputs = True
                                st.rerun()

    with row2[3]: # 🔃 清空選擇
        st.markdown("""
            <style>
            /* 根據按鈕文字選取目標按鈕 */
            button:has(> div:contains("清空選擇")) {
                margin-top: 20px;
            }
            </style>
        """, unsafe_allow_html=True)
        if st.button("🔃 清空選擇"):
            st.session_state.points = []
            st.session_state.nodes = []
            st.session_state.disable_inputs = False  # ✅ 解鎖功能
            st.session_state.has_routed = False
            st.rerun()

    # 統計表格          
    transport_mode = st.session_state.transport_mode
    SPEED = {"機車": 45, "單車": 18, "步行": 5}[transport_mode]

    if st.session_state.has_routed and len(st.session_state.nodes) == 2:
        total = pm25_weight + no2_weight + WBGT_weight
        if total == 0:
            weights = "length"
        else:
            weights = {
                "PM25_expo": pm25_weight / total,
                "NO2_expo": no2_weight / total,
                "WBGT_expo": WBGT_weight / total
            }

        path1, dist1, PM25_acc1, NO2_acc1, WBGT_acc1 = compute_path(G, *st.session_state.nodes, "length")
        path2, dist2, PM25_acc2, NO2_acc2, WBGT_acc2 = compute_path(G, *st.session_state.nodes, weights)

        if path1 is None or path2 is None:
            st.error("⚠️ 找不到可行路徑，請重新設定起點與終點。")
            st.stop()

        dist_km1, dist_km2 = dist1 / 1000, dist2 / 1000
        time_min1 = (dist_km1 / SPEED) * 60
        time_min2 = (dist_km2 / SPEED) * 60

        # 每分鐘曝露量
        rate_pm25_1 = PM25_acc1 / dist1 if dist1 else 0
        rate_pm25_2 = PM25_acc2 / dist2 if dist2 else 0
        rate_no2_1 = NO2_acc1 / dist1 if dist1 else 0
        rate_no2_2 = NO2_acc2 / dist2 if dist2 else 0
        rate_wbgt_1 = WBGT_acc1 / dist1 if dist1 else 0
        rate_wbgt_2 = WBGT_acc2 / dist2 if dist2 else 0

        # 變化率 (%)：以累積值為基礎（最低暴露路徑相較最短路徑）
        improve_pm25 = (PM25_acc2 - PM25_acc1) / PM25_acc1 * 100 if PM25_acc1 else 0
        improve_no2 = (NO2_acc2 - NO2_acc1) / NO2_acc1 * 100 if NO2_acc1 else 0
        improve_wbgt = rate_wbgt_2 - rate_wbgt_1 if WBGT_acc1 else 0
        improve_time = (rate_wbgt_2 - rate_wbgt_1) / time_min1 * 100 if time_min1 else 0

        df = pd.DataFrame({
            "時間/平均暴露": ["預估時間", "PM₂․₅", "NO₂", "氣溫"],
            "🟦最短路徑": [round(time_min1, 2), round(rate_pm25_1, 2), round(rate_no2_1, 2), round(rate_wbgt_1, 2)],
            "🟩舒適路徑": [round(time_min2, 2), round(rate_pm25_2, 2), round(rate_no2_2, 2), round(rate_wbgt_2, 2)],
            "變化率": [
                f"{round(improve_time, 2)}%",
                f"{round(improve_pm25, 2)}%",
                f"{round(improve_no2, 2)}%",
                f"{round(improve_wbgt, 2)}°C"
            ]
        })

        # st.markdown("""
        #     <div style='
        #         font-family: "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", sans-serif !important;
        #         font-size: 20px;
        #         font-weight: 600;
        #         color: #444444;
        #         text-align: center;
        #         margin-top: 0.5em;
        #     '>
        #         統計表格
        #     </div>
        # """, unsafe_allow_html=True)

        st.markdown(
            f"<div class='table-wrapper'>{df.to_html(index=False, classes='centered-table', border=0)}</div>",
            unsafe_allow_html=True
        )

        # 加入 CSS：保留圓角邊框、移除內部格線、維持白字與透明背景（整體縮小一點）
        st.markdown("""
                <style>
                .table-wrapper {
                    width: 90%;
                    margin: auto;
                    border-radius: 12px;
                    border: 1px solid #ccc;
                    overflow: hidden;
                }
                .centered-table {
                    font-size: 16px;  /* 原本是 18px，改小一點 */
                    text-align: center;
                    width: 100%;
                    border-collapse: collapse;
                    font-family: "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", sans-serif !important;
                    background-color: transparent;
                }
                .centered-table th, .centered-table td {
                    padding: 8px;  /* 原本是 12px，改小一點 */
                    background-color: transparent;
                    color: black;
                    border: none;
                    text-align: center;
                }
                .centered-table th {
                    font-weight: bold;
                    color: black;
                    text-align: center;
                }
                .centered-table tr:hover {
                    background-color: transparent !important;
                }
                </style>
            """, unsafe_allow_html=True)

        # 加上灰色小字說明計算目標和單位
        st.markdown("""
            <div style='
                font-size: 12px;
                color: #888888;
                font-family: "Noto Sans TC", "Microsoft JhengHei", sans-serif;
                margin-top: 6px;
                text-align: center;
            '>
                表格數值為每公尺通勤距離下的平均暴露濃度<br>
                單位分別為 分鐘 (時間)、μg/m³（PM₂․₅）、ppb（NO₂）與 °C（氣溫）
            </div>
        """, unsafe_allow_html=True)


with col3:
    # 操作說明
    st.markdown("""
    <style>
    /* expander 整體外框（包含標題區） */
    div.streamlit-expander {
        background-color: #cccccc !important;  /* ✅ 改成你想要的底色 */
        border-radius: 10px !important;
    }

    /* expander 標題列 */
    div.streamlit-expanderHeader {
        font-size: 20px;
        font-weight: 700;
        color: black;
        text-align: center;
        font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)
    with st.expander("🛠️ 操作說明"):
        st.markdown("""
            <div style="
                background-color: #eeeeee;
                padding: 16px;
                border-radius: 10px;
                font-family: 'Noto Sans TC', 'PingFang TC', 'Microsoft JhengHei', sans-serif;
                font-size: 16px;
                color: #444444;
                line-height: 1.6;
            ">
            🟢 輸入起點與終點地址（或點選地圖設定起終點）<br>
            🚘 選擇交通方式：機車、單車或步行<br>
            🧭 點選「路徑解算」：計算兩種路徑（最短/最低暴露），顯示統計表格<br>
            ✅ 點選「空汙疊圖」可查看PM2.5濃度背景圖層
            </div>
        """, unsafe_allow_html=True)

    map_row = st.columns([2, 9])
    
    with map_row[0]:

        # 🟣 空汙疊圖選擇器（改為 radio button）
        st.markdown("""
            <style>
            .overlay-radio .stRadio > div {
                display: flex;
                flex-direction: column;
                gap: 4px;
                font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif;
                font-size: 14px;
            }
            .overlay-radio label {
                font-weight: 600;
            }
            </style>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown("""
                <style>
                .overlay-radio .stRadio > div {
                    display: flex;
                    flex-direction: column;
                    gap: 0px;
                    font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif;
                    font-size: 14px;
                }
                .overlay-radio label {
                    font-weight: 600;
                }
                </style>
            """, unsafe_allow_html=True)

            # 加上標題「圖層」
            st.markdown("""
                <div class="select-label-box">疊加圖層</div>
            """, unsafe_allow_html=True)

            # radio 元件（不顯示 label）
            st.markdown('<div class="overlay-radio">', unsafe_allow_html=True)
            overlay_option = st.radio(
                label="",
                options=["無", "PM₂.₅", "NO₂", "氣溫"],
                index=0,
                key="active_overlay_radio"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # 更新 session_state 對應疊圖層狀態
        if overlay_option == "無":
            st.session_state.pop("active_overlay", None)
        else:
            st.session_state.active_overlay = overlay_option

        st.markdown("""
            <style>
            .full-width-button {
                width: 100%;
                font-size: 14px !important;
                padding: 8px 0 !important;
                margin-bottom: 10px;
                text-align: center;
                font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif;
                font-weight: 600;
            }
            .legend-wrapper {
                margin-top: 16px;
                text-align: center;
                width: 100%;
            }
            .legend-label {
                font-size: 14px;
                font-weight: 600;
                font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif;
                margin: 6px auto;
                padding: 10px 0;
                border-radius: 8px;
                background-color: #eeeeee;
                display: block;
                width: 100%;
                line-height: 1.4;
            }
            </style>
        """, unsafe_allow_html=True)

        ################
        st.markdown("</div>", unsafe_allow_html=True)  # 關掉 transport-wrapper


        # 圖例：不可點擊的樣式展示（縮小空白）
        st.markdown("""
            <div class="legend-wrapper">
                <div class="legend-label">🟦<br>最短路徑</div>
                <div class="legend-label">🟩<br>舒適路徑</div>
            </div>
        """, unsafe_allow_html=True)




    with map_row[1]:


        m = folium.Map(location=map_center, zoom_start=13, control_scale=True)
        m.add_child(DisableDoubleClickZoom())

        for i, pt in enumerate(st.session_state.points):
            label = "起點" if i == 0 else "終點"
            color = "green" if i == 0 else "red"
            folium.Marker(location=pt, tooltip=label, icon=folium.Icon(color=color)).add_to(m)

        if st.session_state.has_routed and len(st.session_state.nodes) == 2:
            for path, color, label in [
                (compute_path(G, *st.session_state.nodes, "length")[0], "blue", "最短路徑"),
                (compute_path(G, *st.session_state.nodes, weights)[0], "#00d26a", "最低暴露路徑")
            ]:
                for u, v in zip(path[:-1], path[1:]):
                    edge_data = G.get_edge_data(u, v)
                    if edge_data:
                        for d in edge_data.values():
                            geom = d.get("attr_dict", {}).get("geometry")
                            if geom:
                                coords = [(lat, lon) for lon, lat in geom.coords]
                                folium.PolyLine(coords, color=color, weight=4, tooltip=label).add_to(m)
                            else:
                                pt1 = G.nodes[u]["latlon"]
                                pt2 = G.nodes[v]["latlon"]
                                folium.PolyLine([pt1, pt2], color=color, weight=4, tooltip=label).add_to(m)

        # 加入 PM2.5、NO2、WBGT 疊圖層（PNG）
        from folium.raster_layers import ImageOverlay
        import base64

        overlay_options = { # 圖片路徑及經緯度
            "PM₂.₅": {
                "path": "data/PM25_全台.png",
                "bounds_twd97": {
                    "left": 147522.218791,
                    "right": 351672.218791,
                    "bottom": 2422004.773102,
                    "top": 2919204.773102
                }
            },
            "NO₂": {
                "path": "data/NO2_全台.png",
                "bounds_twd97": {
                    "left": 147522.218791,
                    "right": 351672.218791,
                    "bottom": 2422004.773102,
                    "top": 2919204.773102
                }
            },
            "氣溫": {
                "path": "data/WBGT_全台.png",
                "bounds_twd97": {
                    "left": 147522.218800,
                    "right": 351672.218800,
                    "bottom": 2422004.773100,
                    "top": 2813154.773100
                }
            }
        }

        # 更新狀態
        if st.session_state.get("active_overlay") == "PM2.5":
            st.session_state.active_overlay = "PM₂.₅"
        if st.session_state.get("active_overlay") == "NO2":
            st.session_state.active_overlay = "NO₂"
        if st.session_state.get("active_overlay") == "氣溫":
            st.session_state.active_overlay = "氣溫"

        # 顯示對應疊圖層
        if "active_overlay" in st.session_state:
            selected = st.session_state.active_overlay
            if selected in overlay_options:
                info = overlay_options[selected]
                bounds = info["bounds_twd97"]
                transformer = Transformer.from_crs("EPSG:3826", "EPSG:4326", always_xy=True)
                left_lon, bottom_lat = transformer.transform(bounds["left"], bounds["bottom"])
                right_lon, top_lat = transformer.transform(bounds["right"], bounds["top"])
                image_bounds = [[bottom_lat, left_lon], [top_lat, right_lon]]

                # 轉 base64
                with open(info["path"], "rb") as f:
                    png_base64 = base64.b64encode(f.read()).decode("utf-8")
                image_url = f"data:image/png;base64,{png_base64}"

                ImageOverlay(
                    image=image_url,
                    bounds=image_bounds,
                    opacity=0.5,
                    interactive=False,
                    cross_origin=False,
                    zindex=1,
                ).add_to(m)


        st_data = st_folium(m, width=600, height=600)

        if not st.session_state.disable_inputs and st_data and st_data.get("last_clicked"):
            latlon = [st_data["last_clicked"]["lat"], st_data["last_clicked"]["lng"]]
            nearest_node = find_nearest_node(G, *latlon)
            if nearest_node:
                lat_, lon_ = G.nodes[nearest_node]["latlon"]
                st.session_state.nodes.append(nearest_node)
                st.session_state.points.append([lat_, lon_])

                # 🔄 加上這段：反查地址並自動填入
                address = reverse_geocode(lat_, lon_)
                if len(st.session_state.points) == 1:
                    st.session_state["set_start_address"] = address
                elif len(st.session_state.points) == 2:
                    st.session_state["set_end_address"] = address

                st.rerun()
            else:
                st.warning("⚠️ 點的位置離路網太遠，請靠近道路再試一次。")

# footer
import streamlit as st
import base64

# 讀取圖片並轉 base64 字串
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

logo_MOE_base64 = image_to_base64("logo/環境部.jpg")
logo_NCKU_base64 = image_to_base64("logo/成大_白色水平.jpg")
logo_GEH_base64 = image_to_base64("logo/實驗室_紅色長方形.jpg")

# 放進 HTML 中
st.markdown(f"""
    <hr style="margin-top: 40px; margin-bottom: 10px; border: none; border-top: 1px solid #ccc;" />

    <div style="text-align: center; font-size: 13px; color: #666; font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif;">
        <p style="margin-bottom: 4px;">
            © 2025 許家瑋 林侑萱 楊蜹畛｜國立成功大學 測量及空間資訊學系<br>
            聯絡信箱：<a href="mailto:p68111509@gs.ncku.edu.tw">p68111509@gs.ncku.edu.tw</a>｜GitHub 專案：<a href="https://github.com/p68111509/Health-routing_Taichung" target="_blank">Health-routing_Taichung</a>
        </p>
        <p style="margin-top: 6px; margin-bottom: 10px;">
            部分空氣汙染空間資訊參考自環境部公開資料
        </p>
        <p style="font-size: 12px; color: #888; margin-top: 0px;">
            本系統僅供展示與研究用途，禁止未經授權之下載、修改、或商業使用。<br>
            所有原始碼、資料與介面設計，皆為作者智慧財產，保留所有權利。
        </p>
        <img src="{logo_NCKU_base64}" alt="College logo" width="180" style="margin-bottom: 10px;">
        <img src="{logo_GEH_base64}" alt="lab logo" width="180" style="margin-bottom: 10px;"><br>
    </div>
""", unsafe_allow_html=True)

# <img src="{logo_MOE_base64}" alt="MOE logo" width="90" style="margin-bottom: 10px;">
#  © 2025 許家瑋 林祐如｜國立成功大學 測量及空間資訊學系｜指導老師：吳治達 教授
# 聯絡信箱：<a href="mailto:p68111509@gs.ncku.edu.tw">p68111509@gs.ncku.edu.tw</a>｜GitHub 專案： <a href="https://github.com/p68111509/low-exposure-routing_demo" target="_blank">low-exposure-routing_demo</a>
