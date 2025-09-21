// 외계 행성 탐사 시뮬레이션 - 시선 속도 방법
// 별과 행성의 공전, 도플러 효과, 흡수 스펙트럼 변화를 시뮬레이션

class ExoplanetSimulation {
    constructor() {
        // 물리 상수
        this.G = 6.67430e-11; // 중력 상수 (m³/kg/s²)
        this.AU = 1.496e11; // 천문단위 (m)
        this.M_SUN = 1.989e30; // 태양 질량 (kg)
        this.M_EARTH = 5.972e24; // 지구 질량 (kg)
        this.C = 299792458; // 빛의 속도 (m/s)
        
        // 시뮬레이션 변수
        this.time = 0;
        this.dt = 86400; // 시간 간격 (1일)
        this.isPlaying = true;
        this.simulationSpeed = 1.0;
        
        // 물리량 (기본값)
        this.star_mass = 1.0; // 태양 질량 단위
        this.planet_mass = 1.0; // 지구 질량 단위
        this.semi_major_axis = 1.0; // AU
        this.eccentricity = 0.0;
        this.inclination = 60; // 도
        this.true_anomaly = 0; // 현재 위치
        
        // 궤도 계산 변수
        this.orbital_period = 0;
        this.mean_motion = 0;
        this.barycenter_x = 0;
        this.barycenter_y = 0;
        
        // 별과 행성의 위치
        this.star_x = 0;
        this.star_y = 0;
        this.planet_x = 0;
        this.planet_y = 0;
        
        // 시선 속도 관련
        this.radial_velocity = 0; // km/s
        this.doppler_shift = 0; // nm
        this.wavelength_shift = 0; // nm
        this.reference_wavelength = 656.3; // Hα선 (nm)
        
        // Canvas 요소들
        this.orbitCanvas = document.getElementById('orbitCanvas');
        this.orbitCtx = this.orbitCanvas.getContext('2d');
        this.spectrumCanvas = document.getElementById('spectrumCanvas');
        this.spectrumCtx = this.spectrumCanvas.getContext('2d');
        
        // UI 요소들
        this.initializeUI();
        this.calculateOrbitalParameters();
        this.startAnimation();
    }
    
    // UI 초기화 및 이벤트 리스너 설정
    initializeUI() {
        // 슬라이더 이벤트 리스너
        document.getElementById('starMass').addEventListener('input', (e) => {
            this.star_mass = parseFloat(e.target.value);
            document.getElementById('starMassValue').textContent = this.star_mass.toFixed(1);
            this.calculateOrbitalParameters();
        });
        
        document.getElementById('planetMass').addEventListener('input', (e) => {
            this.planet_mass = parseFloat(e.target.value);
            document.getElementById('planetMassValue').textContent = this.planet_mass.toFixed(1);
            this.calculateOrbitalParameters();
        });
        
        document.getElementById('semiMajorAxis').addEventListener('input', (e) => {
            this.semi_major_axis = parseFloat(e.target.value);
            document.getElementById('semiMajorAxisValue').textContent = this.semi_major_axis.toFixed(1);
            this.calculateOrbitalParameters();
        });
        
        document.getElementById('eccentricity').addEventListener('input', (e) => {
            this.eccentricity = parseFloat(e.target.value);
            document.getElementById('eccentricityValue').textContent = this.eccentricity.toFixed(2);
            this.calculateOrbitalParameters();
        });
        
        document.getElementById('inclination').addEventListener('input', (e) => {
            this.inclination = parseFloat(e.target.value);
            document.getElementById('inclinationValue').textContent = this.inclination;
            this.calculateOrbitalParameters();
        });
        
        document.getElementById('simulationSpeed').addEventListener('input', (e) => {
            this.simulationSpeed = parseFloat(e.target.value);
            document.getElementById('simulationSpeedValue').textContent = this.simulationSpeed.toFixed(1);
        });
        
        // 버튼 이벤트 리스너
        document.getElementById('playPauseBtn').addEventListener('click', () => {
            this.togglePlayPause();
        });
        
        document.getElementById('resetBtn').addEventListener('click', () => {
            this.resetSimulation();
        });
    }
    
    // 궤도 매개변수 계산
    calculateOrbitalParameters() {
        const a = this.semi_major_axis * this.AU; // 장반축 (m)
        const M_total = (this.star_mass * this.M_SUN + this.planet_mass * this.M_EARTH); // 총 질량
        
        // 케플러 제3법칙으로 공전 주기 계산
        this.orbital_period = 2 * Math.PI * Math.sqrt(a * a * a / (this.G * M_total));
        this.mean_motion = 2 * Math.PI / this.orbital_period;
        
        // 질량 중심 계산
        const total_mass = this.star_mass * this.M_SUN + this.planet_mass * this.M_EARTH;
        const mass_ratio = (this.planet_mass * this.M_EARTH) / total_mass;
        this.barycenter_x = mass_ratio * a;
        this.barycenter_y = 0;
    }
    
    // 현재 시간에서의 위치 계산
    calculatePositions() {
        const a = this.semi_major_axis * this.AU;
        const e = this.eccentricity;
        const i = this.inclination * Math.PI / 180; // 라디안으로 변환
        
        // 평균 이상치 계산
        const mean_anomaly = this.mean_motion * this.time;
        
        // 케플러 방정식으로 진근점 이각 계산 (간단한 근사)
        let E = mean_anomaly; // 이심률이 작을 때의 근사
        if (e > 0.1) {
            // 뉴턴-랩슨 방법으로 정확한 계산
            for (let i = 0; i < 5; i++) {
                E = E - (E - e * Math.sin(E) - mean_anomaly) / (1 - e * Math.cos(E));
            }
        }
        
        const true_anomaly = 2 * Math.atan2(Math.sqrt(1 + e) * Math.sin(E/2), Math.sqrt(1 - e) * Math.cos(E/2));
        
        // 궤도면에서의 위치
        const r = a * (1 - e * e) / (1 + e * Math.cos(true_anomaly));
        const x_orbit = r * Math.cos(true_anomaly);
        const y_orbit = r * Math.sin(true_anomaly);
        
        // 질량 중심을 기준으로 별과 행성의 위치 계산
        const total_mass = this.star_mass * this.M_SUN + this.planet_mass * this.M_EARTH;
        const star_mass_ratio = (this.star_mass * this.M_SUN) / total_mass;
        const planet_mass_ratio = (this.planet_mass * this.M_EARTH) / total_mass;
        
        // 별의 위치 (질량 중심에서 멀리)
        this.star_x = -planet_mass_ratio * x_orbit;
        this.star_y = -planet_mass_ratio * y_orbit;
        
        // 행성의 위치 (질량 중심에서 멀리)
        this.planet_x = star_mass_ratio * x_orbit;
        this.planet_y = star_mass_ratio * y_orbit;
        
        // 시선 속도 계산 (지구에서 관측하는 방향으로의 속도 성분)
        const v_orbit = Math.sqrt(this.G * (this.star_mass * this.M_SUN + this.planet_mass * this.M_EARTH) * 
                                 (2/r - 1/a)) / 1000; // km/s로 변환
        
        // 궤도면에서의 속도 성분
        const vx_orbit = -v_orbit * Math.sin(true_anomaly);
        const vy_orbit = v_orbit * (e + Math.cos(true_anomaly));
        
        // 지구에서 관측하는 시선 속도 (궤도 경사각 고려)
        this.radial_velocity = -star_mass_ratio * vx_orbit * Math.sin(i);
        
        // 도플러 효과 계산
        this.doppler_shift = this.radial_velocity * this.reference_wavelength / this.C * 1e9; // nm
        this.wavelength_shift = this.doppler_shift;
    }
    
    // 궤도 그리기
    drawOrbit() {
        const ctx = this.orbitCtx;
        const width = this.orbitCanvas.width;
        const height = this.orbitCanvas.height;
        
        // 캔버스 초기화
        ctx.fillStyle = '#0c0c0c';
        ctx.fillRect(0, 0, width, height);
        
        // 배경 그리드
        ctx.strokeStyle = 'rgba(79, 172, 254, 0.1)';
        ctx.lineWidth = 1;
        for (let i = 0; i < width; i += 50) {
            ctx.beginPath();
            ctx.moveTo(i, 0);
            ctx.lineTo(i, height);
            ctx.stroke();
        }
        for (let i = 0; i < height; i += 50) {
            ctx.beginPath();
            ctx.moveTo(0, i);
            ctx.lineTo(width, i);
            ctx.stroke();
        }
        
        // 좌표 변환 (중심을 캔버스 중앙으로)
        const centerX = width / 2;
        const centerY = height / 2;
        const scale = 150; // 픽셀/AU
        
        // 궤도 그리기
        ctx.strokeStyle = 'rgba(79, 172, 254, 0.5)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        const a = this.semi_major_axis;
        const e = this.eccentricity;
        const b = a * Math.sqrt(1 - e * e);
        
        for (let angle = 0; angle < 2 * Math.PI; angle += 0.01) {
            const r = a * (1 - e * e) / (1 + e * Math.cos(angle));
            const x = r * Math.cos(angle);
            const y = r * Math.sin(angle);
            
            const pixelX = centerX + x * scale;
            const pixelY = centerY + y * scale;
            
            if (angle === 0) {
                ctx.moveTo(pixelX, pixelY);
            } else {
                ctx.lineTo(pixelX, pixelY);
            }
        }
        ctx.stroke();
        
        // 질량 중심 표시
        ctx.fillStyle = '#ffd700';
        ctx.beginPath();
        ctx.arc(centerX + this.barycenter_x * scale, centerY + this.barycenter_y * scale, 3, 0, 2 * Math.PI);
        ctx.fill();
        
        // 별 그리기
        const starX = centerX + this.star_x * scale;
        const starY = centerY + this.star_y * scale;
        
        // 별의 글로우 효과
        const starGradient = ctx.createRadialGradient(starX, starY, 0, starX, starY, 20);
        starGradient.addColorStop(0, 'rgba(255, 255, 255, 0.8)');
        starGradient.addColorStop(0.5, 'rgba(79, 172, 254, 0.4)');
        starGradient.addColorStop(1, 'rgba(79, 172, 254, 0)');
        
        ctx.fillStyle = starGradient;
        ctx.beginPath();
        ctx.arc(starX, starY, 20, 0, 2 * Math.PI);
        ctx.fill();
        
        // 별의 중심
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(starX, starY, 8, 0, 2 * Math.PI);
        ctx.fill();
        
        // 행성 그리기
        const planetX = centerX + this.planet_x * scale;
        const planetY = centerY + this.planet_y * scale;
        
        ctx.fillStyle = '#4facfe';
        ctx.beginPath();
        ctx.arc(planetX, planetY, 4, 0, 2 * Math.PI);
        ctx.fill();
        
        // 궤도 경로 표시
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(starX, starY);
        ctx.lineTo(planetX, planetY);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // 지구 관측자 표시
        ctx.fillStyle = '#00ff00';
        ctx.beginPath();
        ctx.arc(width - 50, height - 50, 3, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = '#ffffff';
        ctx.font = '12px Arial';
        ctx.fillText('지구', width - 45, height - 45);
    }
    
    // 흡수 스펙트럼 그리기
    drawSpectrum() {
        const ctx = this.spectrumCtx;
        const width = this.spectrumCanvas.width;
        const height = this.spectrumCanvas.height;
        
        // 캔버스 초기화
        ctx.fillStyle = '#0c0c0c';
        ctx.fillRect(0, 0, width, height);
        
        // 기준 파장선
        const centerX = width / 2;
        const centerY = height / 2;
        
        // 스펙트럼 배경
        const spectrumGradient = ctx.createLinearGradient(0, 0, width, 0);
        spectrumGradient.addColorStop(0, '#ff0000');
        spectrumGradient.addColorStop(0.5, '#ffff00');
        spectrumGradient.addColorStop(1, '#0000ff');
        
        ctx.fillStyle = spectrumGradient;
        ctx.fillRect(50, centerY - 20, width - 100, 40);
        
        // 기준 파장선 (Hα)
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(centerX, centerY - 30);
        ctx.lineTo(centerX, centerY + 30);
        ctx.stroke();
        
        // 도플러 시프트된 파장선
        const shiftPixels = (this.wavelength_shift / 10) * 50; // 10nm당 50픽셀
        const shiftedX = centerX + shiftPixels;
        
        if (this.wavelength_shift > 0) {
            // 적색편이 (빨간색)
            ctx.strokeStyle = '#ff6b6b';
        } else {
            // 청색편이 (파란색)
            ctx.strokeStyle = '#4facfe';
        }
        
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(shiftedX, centerY - 30);
        ctx.lineTo(shiftedX, centerY + 30);
        ctx.stroke();
        
        // 화살표로 이동 방향 표시
        ctx.fillStyle = ctx.strokeStyle;
        ctx.beginPath();
        if (this.wavelength_shift > 0) {
            // 오른쪽 화살표 (적색편이)
            ctx.moveTo(shiftedX + 10, centerY);
            ctx.lineTo(shiftedX, centerY - 5);
            ctx.lineTo(shiftedX, centerY + 5);
        } else {
            // 왼쪽 화살표 (청색편이)
            ctx.moveTo(shiftedX - 10, centerY);
            ctx.lineTo(shiftedX, centerY - 5);
            ctx.lineTo(shiftedX, centerY + 5);
        }
        ctx.closePath();
        ctx.fill();
        
        // 파장 표시
        ctx.fillStyle = '#ffffff';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('656.3 nm (Hα)', centerX, centerY + 50);
        ctx.fillText(`${(this.reference_wavelength + this.wavelength_shift).toFixed(2)} nm`, shiftedX, centerY + 70);
        
        // 도플러 시프트 정보
        ctx.textAlign = 'left';
        ctx.fillText(`도플러 시프트: ${this.wavelength_shift.toFixed(3)} nm`, 20, 30);
        ctx.fillText(`시선 속도: ${this.radial_velocity.toFixed(2)} km/s`, 20, 50);
    }
    
    // UI 업데이트
    updateUI() {
        document.getElementById('radialVelocity').textContent = this.radial_velocity.toFixed(2);
        document.getElementById('dopplerShift').textContent = this.doppler_shift.toFixed(3);
        document.getElementById('wavelength').textContent = this.reference_wavelength.toFixed(1);
        document.getElementById('wavelengthShift').textContent = this.wavelength_shift.toFixed(3);
    }
    
    // 애니메이션 루프
    animate() {
        if (this.isPlaying) {
            this.time += this.dt * this.simulationSpeed;
            this.calculatePositions();
        }
        
        this.drawOrbit();
        this.drawSpectrum();
        this.updateUI();
        
        requestAnimationFrame(() => this.animate());
    }
    
    // 애니메이션 시작
    startAnimation() {
        this.animate();
    }
    
    // 재생/일시정지 토글
    togglePlayPause() {
        this.isPlaying = !this.isPlaying;
        const btn = document.getElementById('playPauseBtn');
        btn.textContent = this.isPlaying ? '일시정지' : '재생';
    }
    
    // 시뮬레이션 초기화
    resetSimulation() {
        this.time = 0;
        this.isPlaying = true;
        document.getElementById('playPauseBtn').textContent = '일시정지';
        this.calculatePositions();
    }
}

// 페이지 로드 시 시뮬레이션 시작
document.addEventListener('DOMContentLoaded', () => {
    new ExoplanetSimulation();
});
