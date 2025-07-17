import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd
import io
import base64
from matplotlib.patches import Circle
from matplotlib.widgets import Button
import plotly.graph_objects as go
import plotly.express as px

class WaferMapping:
    def __init__(self):
        self.coordinates = np.array([
            [1, 0, 0], [2, 0, 49], [3, -49, 0], [4, 0, -49], [5, 49, 0],
            [6, 0, 98], [7, -69.3, 69.3], [8, -98, 0], [9, -69.3, -69.3], [10, 0, -98],
            [11, 69.3, -69.3], [12, 98, 0], [13, 69.3, 69.3], [14, 0, 147], [15, -73.5, 127.31],
            [16, -127.31, 73.5], [17, -147.0, 0], [18, -127.31, -73.5], [19, -73.5, -127.31], [20, 0, -147],
            [21, 73.5, -127.31], [22, 127.31, -73.5], [23, 147, 0], [24, 127.31, 73.5], [25, 73.5, 127.31]
        ])
        
        self.wafer_radius = 150
        self.grid_resolution = 1
        self.interpolation_method = 'cubic'
        self.colormap = 'jet'

    def create_wafer_map(self, side1_angle, side2_angle, side1_data, side2_data):
        n_points = len(self.coordinates)
        data = np.zeros((n_points, 5))
        data[:, 0:3] = self.coordinates
        data[:, 3] = side1_data
        data[:, 4] = side2_data

        # 원본 좌표
        X = data[:, 1]
        Y = data[:, 2]
        Z = data[:, 3]
        Z1 = data[:, 4]

        # Side1 회전 매트릭스
        side1_rad = np.deg2rad(side1_angle)
        R1 = np.array([
            [np.cos(side1_rad), -np.sin(side1_rad)],
            [np.sin(side1_rad), np.cos(side1_rad)]
        ])
        
        # Side2 회전 매트릭스
        side2_rad = np.deg2rad(side2_angle)
        R2 = np.array([
            [np.cos(side2_rad), -np.sin(side2_rad)],
            [np.sin(side2_rad), np.cos(side2_rad)]
        ])

        # Side1 회전 적용
        XY_side1 = R1 @ np.vstack([X, Y])
        X_rot_side1 = XY_side1[0, :]
        Y_rot_side1 = XY_side1[1, :]
        
        # Side2 회전 적용
        XY_side2 = R2 @ np.vstack([X, Y])
        X_rot_side2 = XY_side2[0, :]
        Y_rot_side2 = XY_side2[1, :]

        # 그리드 생성
        x_range = np.arange(-self.wafer_radius, self.wafer_radius + 1, self.grid_resolution)
        y_range = np.arange(-self.wafer_radius, self.wafer_radius + 1, self.grid_resolution)
        xq, yq = np.meshgrid(x_range, y_range)

        # 보간
        zq_side1 = griddata((X_rot_side1, Y_rot_side1), Z, (xq, yq), method=self.interpolation_method)
        zq_side2 = griddata((X_rot_side2, Y_rot_side2), Z1, (xq, yq), method=self.interpolation_method)

        # 웨이퍼 영역 마스크
        mask = np.sqrt(xq ** 2 + yq ** 2) > self.wafer_radius
        zq_side1[mask] = np.nan
        zq_side2[mask] = np.nan

        return {
            'data': data,
            'X_rot_side1': X_rot_side1, 'Y_rot_side1': Y_rot_side1,
            'X_rot_side2': X_rot_side2, 'Y_rot_side2': Y_rot_side2,
            'xq': xq, 'yq': yq,
            'zq_side1': zq_side1, 'zq_side2': zq_side2,
            'side1_angle': side1_angle, 'side2_angle': side2_angle,
            'Z': Z, 'Z1': Z1
        }

    def plot_wafer_map(self, result):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Side1 플롯
        im1 = ax1.pcolormesh(result['xq'], result['yq'], result['zq_side1'], 
                            shading='auto', cmap=self.colormap)
        plt.colorbar(im1, ax=ax1, label='Thickness (nm)', shrink=0.8)
        
        # 측정점 표시
        ax1.scatter(result['X_rot_side1'], result['Y_rot_side1'], 
                   c='red', s=30, marker='o', zorder=5)
        
        # 번호와 값 표시
        for i, (x, y, z) in enumerate(zip(result['X_rot_side1'], result['Y_rot_side1'], result['Z'])):
            ax1.text(x, y + 8, str(int(result['data'][i, 0])), 
                    ha='center', va='bottom', fontsize=9, color='black', weight='bold')
            ax1.text(x, y - 8, f"{z:.1f}", 
                    ha='center', va='top', fontsize=8, color='black', weight='bold')
        
        ax1.set_title(f'Side1 - {result["side1_angle"]}° Rotation', fontsize=14, weight='bold')
        ax1.set_xlabel('X (mm)', fontsize=12)
        ax1.set_ylabel('Y (mm)', fontsize=12)
        
        # 웨이퍼 경계선
        circle1 = Circle((0, 0), self.wafer_radius, fill=False, edgecolor='black', linewidth=2)
        ax1.add_patch(circle1)
        
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        margin = self.wafer_radius * 0.2
        ax1.set_xlim(-self.wafer_radius - margin, self.wafer_radius + margin)
        ax1.set_ylim(-self.wafer_radius - margin, self.wafer_radius + margin)
        
        # Side2 플롯
        im2 = ax2.pcolormesh(result['xq'], result['yq'], result['zq_side2'], 
                            shading='auto', cmap=self.colormap)
        plt.colorbar(im2, ax=ax2, label='Thickness (nm)', shrink=0.8)
        
        # 측정점 표시
        ax2.scatter(result['X_rot_side2'], result['Y_rot_side2'], 
                   c='red', s=30, marker='o', zorder=5)
        
        # 번호와 값 표시
        for i, (x, y, z1) in enumerate(zip(result['X_rot_side2'], result['Y_rot_side2'], result['Z1'])):
            ax2.text(x, y + 8, str(int(result['data'][i, 0])), 
                    ha='center', va='bottom', fontsize=9, color='black', weight='bold')
            ax2.text(x, y - 8, f"{z1:.1f}", 
                    ha='center', va='top', fontsize=8, color='black', weight='bold')
        
        ax2.set_title(f'Side2 - {result["side2_angle"]}° Rotation', fontsize=14, weight='bold')
        ax2.set_xlabel('X (mm)', fontsize=12)
        ax2.set_ylabel('Y (mm)', fontsize=12)
        
        # 웨이퍼 경계선
        circle2 = Circle((0, 0), self.wafer_radius, fill=False, edgecolor='black', linewidth=2)
        ax2.add_patch(circle2)
        
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-self.wafer_radius - margin, self.wafer_radius + margin)
        ax2.set_ylim(-self.wafer_radius - margin, self.wafer_radius + margin)
        
        plt.tight_layout()
        return fig
    
    def plot_wafer_map_plotly(self, result):
        from plotly.subplots import make_subplots
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                f'Side1 - {result["side1_angle"]}° Rotation',
                f'Side2 - {result["side2_angle"]}° Rotation'
            ],
            horizontal_spacing=0.1
        )
        
        # Side1 히트맵
        fig.add_trace(
            go.Heatmap(
                z=result['zq_side1'],
                x=result['xq'][0, :],
                y=result['yq'][:, 0],
                colorscale='Jet',
                showscale=True,
                colorbar=dict(title="Thickness (nm)", x=0.45),
                name="Side1"
            ),
            row=1, col=1
        )
        
        # Side1 측정점 추가
        fig.add_trace(
            go.Scatter(
                x=result['X_rot_side1'],
                y=result['Y_rot_side1'],
                mode='markers+text',
                marker=dict(color='red', size=8),
                text=[f"{int(result['data'][i, 0])}<br>{result['Z'][i]:.1f}" for i in range(25)],
                textposition="middle center",
                textfont=dict(color='white', size=10),
                name="Side1 Points",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Side2 히트맵
        fig.add_trace(
            go.Heatmap(
                z=result['zq_side2'],
                x=result['xq'][0, :],
                y=result['yq'][:, 0],
                colorscale='Jet',
                showscale=True,
                colorbar=dict(title="Thickness (nm)", x=1.02),
                name="Side2"
            ),
            row=1, col=2
        )
        
        # Side2 측정점 추가
        fig.add_trace(
            go.Scatter(
                x=result['X_rot_side2'],
                y=result['Y_rot_side2'],
                mode='markers+text',
                marker=dict(color='red', size=8),
                text=[f"{int(result['data'][i, 0])}<br>{result['Z1'][i]:.1f}" for i in range(25)],
                textposition="middle center",
                textfont=dict(color='white', size=10),
                name="Side2 Points",
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 웨이퍼 경계선 추가
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = self.wafer_radius * np.cos(theta)
        circle_y = self.wafer_radius * np.sin(theta)
        
        # Side1 경계선
        fig.add_trace(
            go.Scatter(
                x=circle_x,
                y=circle_y,
                mode='lines',
                line=dict(color='black', width=2),
                name="Wafer Boundary",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Side2 경계선
        fig.add_trace(
            go.Scatter(
                x=circle_x,
                y=circle_y,
                mode='lines',
                line=dict(color='black', width=2),
                name="Wafer Boundary",
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 레이아웃 설정
        margin = self.wafer_radius * 0.2
        fig.update_layout(
            title=f"웨이퍼 매핑 결과 (Side1: {result['side1_angle']}°, Side2: {result['side2_angle']}°)",
            height=600,
            showlegend=False
        )
        
        # 축 설정
        fig.update_xaxes(
            title_text="X (mm)",
            range=[-self.wafer_radius - margin, self.wafer_radius + margin],
            scaleanchor="y",
            scaleratio=1
        )
        
        fig.update_yaxes(
            title_text="Y (mm)",
            range=[-self.wafer_radius - margin, self.wafer_radius + margin],
            scaleanchor="x",
            scaleratio=1
        )
        
        return fig

def get_image_download_link(fig, filename):
    """matplotlib 그림을 다운로드 링크로 변환"""
    img = io.BytesIO()
    fig.savefig(img, format='png', dpi=300, bbox_inches='tight')
    img.seek(0)
    b64 = base64.b64encode(img.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">이미지 다운로드</a>'
    return href

def main():
    st.set_page_config(
        page_title="웨이퍼 매핑 도구",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("웨이퍼 매핑 도구")
    st.markdown("---")
    
    # 사이드바 설정
    st.sidebar.header("설정")
    
    # 각도 옵션
    st.sidebar.subheader("각도 옵션")
    angle_options = {
        'LP1': [256.90, 45.0, 60.0],
        'LP2': [256.90, 50.0, 70.0],
        'LP3': [90.294, 80, 90]
    }
    
    # 각도 입력 방식 선택
    angle_mode = st.sidebar.radio(
        "각도 입력 방식",
        ["사전 정의된 옵션", "임의 지정"],
        help="각도를 선택하거나 직접 입력하세요"
    )
    
    if angle_mode == "사전 정의된 옵션":
        # LP 선택
        lp_choice = st.sidebar.selectbox(
            "LP 옵션 선택",
            ["LP1", "LP2", "LP3"],
            help="LP1: [256.90°, 45.0°, 60.0°], LP2: [256.90°, 50.0°, 70.0°], LP3: [90.294°, 80°, 90°]"
        )
        
        # 각도 선택
        angles = angle_options[lp_choice]
        
        side1_angle_index = st.sidebar.selectbox(
            "Side1 각도 선택",
            range(len(angles)),
            format_func=lambda x: f"{angles[x]}°",
            help=f"Side1 각도 옵션: {angles}"
        )
        
        side2_angle_index = st.sidebar.selectbox(
            "Side2 각도 선택",
            range(len(angles)),
            format_func=lambda x: f"{angles[x]}°",
            help=f"Side2 각도 옵션: {angles}"
        )
        
        side1_angle = angles[side1_angle_index]
        side2_angle = angles[side2_angle_index]
        
    else:  # 임의 지정
        # 기본값 각도 설정
        st.sidebar.subheader("기본값 각도 설정")
        default_side1 = st.sidebar.number_input(
            "Side1 기본각도 (도)",
            min_value=-180.0,
            max_value=180.0,
            value=0.0,
            step=0.1,
            help="Side1의 기본 회전 각도를 설정하세요"
        )
        
        default_side2 = st.sidebar.number_input(
            "Side2 기본각도 (도)",
            min_value=-180.0,
            max_value=180.0,
            value=0.0,
            step=0.1,
            help="Side2의 기본 회전 각도를 설정하세요"
        )
        
        # 각도 입력
        st.sidebar.subheader("각도 설정")
        side1_angle = st.sidebar.number_input(
            "Side1 각도 (도)",
            min_value=-180.0,
            max_value=180.0,
            value=default_side1,
            step=0.1,
            help="Side1 회전 각도를 입력하세요"
        )
        
        side2_angle = st.sidebar.number_input(
            "Side2 각도 (도)",
            min_value=-180.0,
            max_value=180.0,
            value=default_side2,
            step=0.1,
            help="Side2 회전 각도를 입력하세요"
        )
    
    
    wm = WaferMapping()
    
    # 메인 영역
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("데이터 입력")
        
        side1_data = None
        side2_data = None
        
        # 데이터 입력 방식 선택
        input_method = st.radio(
            "데이터 입력 방식",
            ["데모 데이터", "직접 입력"],
            help="데모 데이터 또는 직접 입력을 선택하세요"
        )
        
        if input_method == "데모 데이터":
            st.info("데모 데이터를 사용합니다.")
            # 데모 데이터 생성
            np.random.seed(42)
            side1_data = np.random.normal(2500, 50, 25)
            side2_data = np.random.normal(2600, 50, 25)
            
        else:  # 직접 입력
            st.subheader("Side1 두께 데이터")
            side1_text = st.text_area(
                "Side1 값 입력 (25개, 각 줄에 하나씩)",
                height=200,
                placeholder="2500.1\n2501.2\n2502.3\n...",
                help="숫자 데이터를 한 줄에 하나씩 입력하세요"
            )
            
            st.subheader("Side2 두께 데이터")
            side2_text = st.text_area(
                "Side2 값 입력 (25개, 각 줄에 하나씩)",
                height=200,
                placeholder="2600.1\n2601.2\n2602.3\n...",
                help="숫자 데이터를 한 줄에 하나씩 입력하세요"
            )
        
            if side1_text and side2_text:
                try:
                    side1_values = [float(x.strip()) for x in side1_text.strip().split('\n') if x.strip()]
                    side2_values = [float(x.strip()) for x in side2_text.strip().split('\n') if x.strip()]
                    
                    if len(side1_values) == 25 and len(side2_values) == 25:
                        side1_data = np.array(side1_values)
                        side2_data = np.array(side2_values)
                        st.success("데이터 입력 완료!")
                    else:
                        st.error(f"각 면마다 정확히 25개의 값이 필요합니다. (Side1: {len(side1_values)}, Side2: {len(side2_values)})")
                except ValueError:
                    st.error("숫자 형식이 올바르지 않습니다.")
    
    with col2:
        st.header("웨이퍼 맵")
        
        if side1_data is not None and side2_data is not None:
            # 웨이퍼 매핑 실행
            result = wm.create_wafer_map(side1_angle, side2_angle, side1_data, side2_data)
            
            # 시각화 옵션
            viz_option = st.radio(
                "시각화 옵션",
                ["정적 그래프 (Matplotlib)", "인터랙티브 그래프 (Plotly)"],
                horizontal=True
            )
            
            if viz_option == "정적 그래프 (Matplotlib)":
                # 그래프 생성
                fig = wm.plot_wafer_map(result)
                st.pyplot(fig)
                
                # 다운로드 링크
                filename = f"wafer_map_S1_{result['side1_angle']}_S2_{result['side2_angle']}.png"
                download_link = get_image_download_link(fig, filename)
                st.markdown(download_link, unsafe_allow_html=True)
            
            else:
                # Plotly 인터랙티브 그래프
                fig_plotly = wm.plot_wafer_map_plotly(result)
                st.plotly_chart(fig_plotly, use_container_width=True)
            
            # 통계 정보
            st.subheader("통계 정보")
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                st.metric("Side1 평균", f"{np.nanmean(result['zq_side1']):.2f} nm")
                st.metric("Side1 표준편차", f"{np.nanstd(result['zq_side1']):.2f} nm")
                st.metric("Side1 범위", f"{np.nanmin(result['zq_side1']):.1f} ~ {np.nanmax(result['zq_side1']):.1f} nm")
            
            with col_stat2:
                st.metric("Side2 평균", f"{np.nanmean(result['zq_side2']):.2f} nm")
                st.metric("Side2 표준편차", f"{np.nanstd(result['zq_side2']):.2f} nm")
                st.metric("Side2 범위", f"{np.nanmin(result['zq_side2']):.1f} ~ {np.nanmax(result['zq_side2']):.1f} nm")
            
            # 측정점 데이터 테이블
            st.subheader("측정점 정보")
            table_data = []
            for i in range(25):
                table_data.append({
                    'Point': int(result['data'][i, 0]),
                    'X': result['data'][i, 1],
                    'Y': result['data'][i, 2],
                    'Side1': result['Z'][i],
                    'Side2': result['Z1'][i]
                })
            
            df_table = pd.DataFrame(table_data)
            st.dataframe(df_table, use_container_width=True)
            
        else:
            st.info("왼쪽에서 데이터를 입력하면 웨이퍼 맵이 표시됩니다.")
    
    # 푸터
    st.markdown("---")
    st.markdown("**웨이퍼 매핑 도구** | 회사 공유용 버전")

if __name__ == "__main__":
    main()