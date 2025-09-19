// RV (Doppler) Exoplanet Simulator - 웹 버전
// 물리 상수들
const G = 6.67430e-11;        // 중력 상수
const M_SUN = 1.98847e30;     // 태양 질량 (kg)
const M_JUP = 1.89813e27;     // 목성 질량 (kg)
const DAY = 86400.0;          // 하루 (초)
const AU = 1.495978707e11;    // 천문단위 (m)
const C = 299792458.0;        // 빛의 속도 (m/s)
const DOPPLER_SCALE = 1e4;    // 도플러 효과 시각화 스케일

// 시뮬레이터 클래스
class RvSimulator {
    constructor() {
        this.params = {
            M_star_solar: 1.0,
            M_p_jup: 1.0,
            inc_deg: 90.0,
            period_days: 365.25,
            ecc: 0.0,
            omega_deg: 0.0,
            t0_days: 0.0,
            gamma_ms: 0.0,
            base_lambda_nm: 656.28,
            a_au: 1.0,
            kepler_lock: true
        };
        
        this.t_sim = 0.0;
        this.dt = 5.0 * DAY;
        this.running = false;
        this.t_list = [];
        this.rv_list = [];
        
        this.scale_factor = 2000.0;
        this.rv_window_days = 400.0;
        this.schematic = true;
        this.bary_boost = 80.0;
        
        this.updateParams();
    }
    
    updateParams() {
        // 파라미터 정규화
        const Ms_solar = Math.max(1e-12, this.params.M_star_solar);
        const Mp_jup = Math.max(0.0, this.params.M_p_jup);
        const inc_deg = this.params.inc_deg;
        const period_d = Math.max(1e-9, this.params.period_days);
        const ecc_in = this.params.ecc;
        const ecc_clamped = Math.min(Math.max(ecc_in, 0.0), 0.999999);
        
        // 단위 변환
        this.Ms = Ms_solar * M_SUN;
        this.Mp = Mp_jup * M_JUP;
        this.i = inc_deg * Math.PI / 180;  // 라디안으로 변환
        this.e = ecc_clamped;
        this.omega = this.params.omega_deg * Math.PI / 180;
        this.t0 = this.params.t0_days * DAY;
        this.gamma = this.params.gamma_ms;
        this.lam0 = this.params.base_lambda_nm;
        
        // Kepler lock 처리
        if (this.params.kepler_lock) {
            // a → P
            this.a = this.params.a_au * AU;
            this.P = 2 * Math.PI * Math.sqrt(this.a**3 / (G * (this.Ms + this.Mp)));
            this.params.period_days = this.P / DAY;
        } else {
            // P → a
            this.P = period_d * DAY;
            this.a = (G * (this.Ms + this.Mp) * this.P**2 / (4*Math.PI**2)) ** (1/3.0);
            this.params.a_au = this.a / AU;
        }
        
        // 질량비에 따른 별/행성 궤도 반지름
        this.a_star = this.a * this.Mp / (this.Ms + this.Mp);
        this.a_planet = this.a * this.Ms / (this.Ms + this.Mp);
        
        // K(세미진폭) 계산
        const denom = Math.max(1e-16, 1.0 - this.e**2);
        this.K = (((2 * Math.PI * G) / this.P) ** (1/3.0)
                  * (this.Mp * Math.sin(this.i))
                  / ((this.Ms + this.Mp) ** (2/3.0))
                  / Math.sqrt(denom));
    }
    
    meanAnomaly(t) {
        return 2 * Math.PI * ((t - this.t0) % this.P) / this.P;
    }
    
    eccentricAnomaly(M) {
        let E = M < 0.8 ? M : Math.PI;
        for (let i = 0; i < 50; i++) {
            const f = E - this.e * Math.sin(E) - M;
            const fp = 1 - this.e * Math.cos(E);
            const dE = -f / fp;
            E += dE;
            if (Math.abs(dE) < 1e-12) break;
        }
        return E;
    }
    
    trueAnomaly(E) {
        const factor = Math.sqrt((1 + this.e) / (1 - this.e));
        return 2 * Math.atan2(factor * Math.sin(E/2), Math.cos(E/2));
    }
    
    rv(t) {
        // 궤도 위상 계산
        const M = this.meanAnomaly(t);
        const E = this.eccentricAnomaly(M);
        const nu = this.trueAnomaly(E);
        const theta = nu + this.omega;
        
        // 궤도 상수
        const mu = G * (this.Ms + this.Mp);
        const h = Math.sqrt(mu * this.a * (1.0 - this.e**2));
        const v_scale = mu / h;
        
        // 상대속도를 LOS로 투영
        const v_rel_los = -Math.sin(theta) * v_scale;
        
        // 별의 속도
        const v_star_los = -(this.Mp / (this.Ms + this.Mp)) * v_rel_los * Math.sin(this.i);
        
        return v_star_los + this.gamma;
    }
    
    starPlanetPositions(t, scale_factor = 2000.0, schematic = false, bary_boost = 1.0) {
        const M = this.meanAnomaly(t);
        const E = this.eccentricAnomaly(M);
        const nu = this.trueAnomaly(E);
        const theta = nu + this.omega;
        
        // 시각적 질량비 계산
        const mratio_true = this.Mp / this.Ms;
        const mratio_vis = Math.max(1e-16, mratio_true * Math.max(1.0, bary_boost));
        let a_star_vis = this.a * (mratio_vis / (1.0 + mratio_vis));
        let a_planet_vis = this.a * (1.0 / (1.0 + mratio_vis));
        
        if (schematic) {
            // 원 궤도 모식도
            const r_star = a_star_vis * scale_factor;
            const r_planet = a_planet_vis * scale_factor;
            
            const x_s = -r_star * Math.cos(theta) * Math.sin(this.i);
            const y_s = -r_star * Math.sin(theta);
            const x_p = r_planet * Math.cos(theta) * Math.sin(this.i);
            const y_p = r_planet * Math.sin(theta);
            
            return { star: { x: y_s, y: x_s }, planet: { x: y_p, y: x_p } };
        } else {
            // 타원 궤도 (r(ν) 반영)
            const fac = (1.0 - this.e**2) / (1.0 + this.e * Math.cos(nu));
            const r_star = a_star_vis * fac * scale_factor;
            const r_planet = a_planet_vis * fac * scale_factor;
            
            const x_s = -r_star * Math.cos(theta) * Math.sin(this.i);
            const y_s = -r_star * Math.sin(theta);
            const x_p = r_planet * Math.cos(theta) * Math.sin(this.i);
            const y_p = r_planet * Math.sin(theta);
            
            return { star: { x: y_s, y: x_s }, planet: { x: y_p, y: x_p } };
        }
    }
}

// 차트 관리 클래스
class ChartManager {
    constructor() {
        this.rvChart = null;
        this.spectrumChart = null;
        this.orbitChart = null;
        this.initCharts();
    }
    
    initCharts() {
        // RV 차트
        const rvCtx = document.getElementById('rvChart').getContext('2d');
        this.rvChart = new Chart(rvCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'RV (m/s)',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Time (days)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'RV (m/s)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
        
        // 스펙트럼 차트
        const specCtx = document.getElementById('spectrumChart').getContext('2d');
        this.spectrumChart = new Chart(specCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Spectrum',
                    data: [],
                    borderColor: 'rgba(0, 0, 0, 0.8)',
                    backgroundColor: 'rgba(0, 0, 0, 0.3)',
                    tension: 0.1,
                    fill: true,
                    pointRadius: 0,
                    pointHoverRadius: 3,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Wavelength (nm)',
                            color: 'white',
                            font: {
                                size: 12,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: 'rgba(255,255,255,0.3)'
                        },
                        ticks: {
                            color: 'white',
                            font: {
                                size: 10
                            }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Intensity (Relative)',
                            color: 'white',
                            font: {
                                size: 12,
                                weight: 'bold'
                            }
                        },
                        min: 0,
                        max: 1.1,
                        grid: {
                            color: 'rgba(255,255,255,0.3)'
                        },
                        ticks: {
                            color: 'white',
                            font: {
                                size: 10
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0,0,0,0.8)',
                        titleColor: 'white',
                        bodyColor: 'white',
                        callbacks: {
                            label: function(context) {
                                return `λ = ${context.parsed.x.toFixed(3)} nm, I = ${context.parsed.y.toFixed(3)}`;
                            }
                        }
                    }
                }
            }
        });
        
        // 궤도 차트 초기화
        this.initOrbitChart();
    }
    
    initOrbitChart() {
        const canvas = document.getElementById('orbitChart');
        this.orbitCtx = canvas.getContext('2d');
        this.orbitCanvas = canvas;
        
        // 캔버스 크기 설정
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;
        this.orbitCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }
    
    updateRvChart(t_list, rv_list) {
        this.rvChart.data.labels = t_list;
        this.rvChart.data.datasets[0].data = rv_list;
        this.rvChart.update('none');
    }
    
    updateSpectrumChart(lambda_shift, lambda_0) {
        // 흡수 스펙트럼 데이터 생성
        const wavelengths = [];
        const intensities = [];
        const range = 2.0; // nm 범위
        const points = 500;
        
        for (let i = 0; i < points; i++) {
            const wl = lambda_0 - range/2 + (range * i / points);
            wavelengths.push(wl);
            
            // 연속 스펙트럼 (1.0)에서 흡수선을 뺀 형태
            let intensity = 1.0; // 연속 스펙트럼 기본값
            
            // 중심 파장에서 흡수선 (로렌츠 형태 - 더 현실적)
            const gamma = 0.01; // 흡수선 폭 (로렌츠 폭)
            const absorption_depth = 0.85; // 흡수 깊이
            const absorption = absorption_depth * (gamma * gamma) / ((wl - lambda_0) * (wl - lambda_0) + gamma * gamma);
            intensity -= absorption;
            
            // 도플러 이동된 흡수선 (현재 위치)
            const doppler_gamma = 0.008; // 도플러 선은 더 좁게
            const doppler_absorption = absorption_depth * (doppler_gamma * doppler_gamma) / ((wl - lambda_shift) * (wl - lambda_shift) + doppler_gamma * doppler_gamma);
            intensity -= doppler_absorption;
            
            // 추가적인 흡수선들 (더 현실적인 스펙트럼)
            const additional_lines = [
                { center: lambda_0 - 0.08, depth: 0.25, width: 0.005 },
                { center: lambda_0 + 0.08, depth: 0.2, width: 0.005 },
                { center: lambda_0 - 0.25, depth: 0.3, width: 0.008 },
                { center: lambda_0 + 0.25, depth: 0.28, width: 0.008 },
                { center: lambda_0 - 0.5, depth: 0.15, width: 0.01 },
                { center: lambda_0 + 0.5, depth: 0.12, width: 0.01 }
            ];
            
            additional_lines.forEach(line => {
                const additional_absorption = line.depth * (line.width * line.width) / ((wl - line.center) * (wl - line.center) + line.width * line.width);
                intensity -= additional_absorption;
            });
            
            // 최소값 제한
            intensity = Math.max(0.05, intensity);
            intensities.push(intensity);
        }
        
        this.spectrumChart.data.labels = wavelengths;
        this.spectrumChart.data.datasets[0].data = intensities;
        this.spectrumChart.update('none');
        
        // 도플러 이동된 선을 수직선으로 표시
        this.drawDopplerLine(lambda_shift, lambda_0);
    }
    
    drawDopplerLine(lambda_shift, lambda_0) {
        // 스펙트럼 차트에 도플러 이동된 선 그리기
        const chart = this.spectrumChart;
        const ctx = chart.ctx;
        const chartArea = chart.chartArea;
        
        // 기존 선 제거를 위해 차트 다시 그리기
        chart.draw();
        
        // 도플러 이동된 선 그리기
        const x = chart.scales.x.getPixelForValue(lambda_shift);
        const y1 = chartArea.top;
        const y2 = chartArea.bottom;
        
        // 빨간색 점선으로 도플러 이동된 선 표시
        ctx.save();
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 4;
        ctx.setLineDash([8, 4]);
        ctx.beginPath();
        ctx.moveTo(x, y1);
        ctx.lineTo(x, y2);
        ctx.stroke();
        ctx.restore();
        
        // 라벨 그리기 (배경과 함께)
        ctx.save();
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(x - 40, y1 - 25, 80, 20);
        ctx.fillStyle = 'red';
        ctx.font = 'bold 11px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`λ = ${lambda_shift.toFixed(3)} nm`, x, y1 - 10);
        ctx.restore();
        
        // 중심 파장도 표시 (비교용)
        const center_x = chart.scales.x.getPixelForValue(lambda_0);
        ctx.save();
        ctx.strokeStyle = 'yellow';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 2]);
        ctx.beginPath();
        ctx.moveTo(center_x, y1);
        ctx.lineTo(center_x, y2);
        ctx.stroke();
        ctx.restore();
        
        // 중심 파장 라벨
        ctx.save();
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(center_x - 35, y2 + 5, 70, 20);
        ctx.fillStyle = 'yellow';
        ctx.font = 'bold 10px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`λ₀ = ${lambda_0.toFixed(3)} nm`, center_x, y2 + 18);
        ctx.restore();
    }
    
    drawOrbit(simulator) {
        const ctx = this.orbitCtx;
        const canvas = this.orbitCanvas;
        const rect = canvas.getBoundingClientRect();
        
        // 캔버스 클리어
        ctx.clearRect(0, 0, rect.width, rect.height);
        
        // 배경
        ctx.fillStyle = '#f8f9fa';
        ctx.fillRect(0, 0, rect.width, rect.height);
        
        // 중심점 (바리센터)
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        
        // 궤도 경로 그리기
        const nu_arr = [];
        for (let i = 0; i < 400; i++) {
            nu_arr.push(2 * Math.PI * i / 400);
        }
        
        const theta_arr = nu_arr.map(nu => nu + simulator.omega);
        const boost = simulator.bary_boost;
        
        // 시각적 질량비
        const mratio_true = simulator.Mp / simulator.Ms;
        const mratio_vis = Math.max(1e-16, mratio_true * Math.max(1.0, boost));
        const a_star_vis = simulator.a * (mratio_vis / (1.0 + mratio_vis));
        const a_planet_vis = simulator.a * (1.0 / (1.0 + mratio_vis));
        
        let r_s, r_p;
        if (simulator.schematic) {
            r_s = a_star_vis * simulator.scale_factor;
            r_p = a_planet_vis * simulator.scale_factor;
        } else {
            const fac = (1.0 - simulator.e**2) / (1.0 + simulator.e * Math.cos(nu_arr[0]));
            r_s = a_star_vis * fac * simulator.scale_factor;
            r_p = a_planet_vis * fac * simulator.scale_factor;
        }
        
        // 궤도 경로 그리기
        ctx.strokeStyle = 'orange';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        for (let i = 0; i < nu_arr.length; i++) {
            const nu = nu_arr[i];
            const theta = theta_arr[i];
            
            let r_star, r_planet;
            if (simulator.schematic) {
                r_star = a_star_vis * simulator.scale_factor;
                r_planet = a_planet_vis * simulator.scale_factor;
            } else {
                const fac = (1.0 - simulator.e**2) / (1.0 + simulator.e * Math.cos(nu));
                r_star = a_star_vis * fac * simulator.scale_factor;
                r_planet = a_planet_vis * fac * simulator.scale_factor;
            }
            
            const x_s = -r_star * Math.cos(theta) * Math.sin(simulator.i);
            const y_s = -r_star * Math.sin(theta);
            const x_p = r_planet * Math.cos(theta) * Math.sin(simulator.i);
            const y_p = r_planet * Math.sin(theta);
            
            const plot_x_s = centerX + y_s;
            const plot_y_s = centerY + x_s;
            const plot_x_p = centerX + y_p;
            const plot_y_p = centerY + x_p;
            
            if (i === 0) {
                ctx.moveTo(plot_x_s, plot_y_s);
            } else {
                ctx.lineTo(plot_x_s, plot_y_s);
            }
        }
        ctx.stroke();
        
        // 행성 궤도
        ctx.strokeStyle = 'saddlebrown';
        ctx.beginPath();
        for (let i = 0; i < nu_arr.length; i++) {
            const nu = nu_arr[i];
            const theta = theta_arr[i];
            
            let r_planet;
            if (simulator.schematic) {
                r_planet = a_planet_vis * simulator.scale_factor;
            } else {
                const fac = (1.0 - simulator.e**2) / (1.0 + simulator.e * Math.cos(nu));
                r_planet = a_planet_vis * fac * simulator.scale_factor;
            }
            
            const x_p = r_planet * Math.cos(theta) * Math.sin(simulator.i);
            const y_p = r_planet * Math.sin(theta);
            
            const plot_x_p = centerX + y_p;
            const plot_y_p = centerY + x_p;
            
            if (i === 0) {
                ctx.moveTo(plot_x_p, plot_y_p);
            } else {
                ctx.lineTo(plot_x_p, plot_y_p);
            }
        }
        ctx.stroke();
        
        // 바리센터 표시
        ctx.fillStyle = 'black';
        ctx.beginPath();
        ctx.arc(centerX, centerY, 5, 0, 2 * Math.PI);
        ctx.fill();
        
        // 지구 표시
        const earthY = centerY + 100;
        ctx.fillStyle = 'blue';
        ctx.beginPath();
        ctx.arc(centerX, earthY, 8, 0, 2 * Math.PI);
        ctx.fill();
        
        // 지구 라벨
        ctx.fillStyle = 'blue';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Earth', centerX, earthY + 20);
        
        // 현재 위치 표시
        const pos = simulator.starPlanetPositions(simulator.t_sim, simulator.scale_factor, simulator.schematic, simulator.bary_boost);
        
        // 별 위치
        ctx.fillStyle = 'gold';
        ctx.beginPath();
        ctx.arc(centerX + pos.star.x, centerY + pos.star.y, 6, 0, 2 * Math.PI);
        ctx.fill();
        
        // 행성 위치
        ctx.fillStyle = 'tab:blue';
        ctx.beginPath();
        ctx.arc(centerX + pos.planet.x, centerY + pos.planet.y, 4, 0, 2 * Math.PI);
        ctx.fill();
    }
}

// 전역 변수
let simulator;
let chartManager;
let animationId;

// Chart.js 로딩 대기 함수
function waitForChart() {
    return new Promise((resolve) => {
        if (typeof Chart !== 'undefined') {
            resolve();
        } else {
            const checkChart = setInterval(() => {
                if (typeof Chart !== 'undefined') {
                    clearInterval(checkChart);
                    resolve();
                }
            }, 100);
        }
    });
}

// 초기화
document.addEventListener('DOMContentLoaded', async function() {
    // Chart.js 로딩 대기
    await waitForChart();
    
    simulator = new RvSimulator();
    chartManager = new ChartManager();
    
    // 슬라이더 값 표시
    const baryBoostSlider = document.getElementById('bary_boost');
    const baryBoostValue = document.getElementById('bary_boost_value');
    
    baryBoostSlider.addEventListener('input', function() {
        baryBoostValue.textContent = this.value;
        simulator.bary_boost = parseFloat(this.value);
    });
    
    // 체크박스 이벤트
    document.getElementById('schematic').addEventListener('change', function() {
        simulator.schematic = this.checked;
    });
    
    // 초기 차트 업데이트
    updateCharts();
    
    console.log('RV Exoplanet Simulator 초기화 완료');
});

// 파라미터 적용
function applyParams() {
    simulator.params.M_star_solar = parseFloat(document.getElementById('M_star_solar').value);
    simulator.params.M_p_jup = parseFloat(document.getElementById('M_p_jup').value);
    simulator.params.inc_deg = parseFloat(document.getElementById('inc_deg').value);
    simulator.params.period_days = parseFloat(document.getElementById('period_days').value);
    simulator.params.a_au = parseFloat(document.getElementById('a_au').value);
    simulator.params.ecc = parseFloat(document.getElementById('ecc').value);
    simulator.params.omega_deg = parseFloat(document.getElementById('omega_deg').value);
    simulator.params.t0_days = parseFloat(document.getElementById('t0_days').value);
    simulator.params.gamma_ms = parseFloat(document.getElementById('gamma_ms').value);
    simulator.params.base_lambda_nm = parseFloat(document.getElementById('base_lambda_nm').value);
    simulator.params.kepler_lock = document.getElementById('kepler_lock').checked;
    
    simulator.updateParams();
    
    // UI 동기화
    if (simulator.params.kepler_lock) {
        document.getElementById('period_days').value = simulator.params.period_days.toFixed(2);
    }
    document.getElementById('a_au').value = simulator.params.a_au.toFixed(2);
    
    reset();
}

// 재생/정지 토글
function toggleRun() {
    if (!simulator || !chartManager) {
        console.error('시뮬레이터가 초기화되지 않았습니다.');
        return;
    }
    
    simulator.running = !simulator.running;
    const btn = document.getElementById('toggleBtn');
    btn.textContent = simulator.running ? '⏸ 정지' : '▶ 재생';
    
    if (simulator.running) {
        console.log('애니메이션 시작');
        animate();
    } else {
        console.log('애니메이션 정지');
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
    }
}

// 한 스텝 실행
function stepOnce() {
    simulator.t_sim += simulator.dt;
    updateCharts();
}

// 초기화
function reset() {
    simulator.t_sim = 0.0;
    simulator.t_list = [];
    simulator.rv_list = [];
    updateCharts();
}

// 애니메이션 루프
function animate() {
    if (!simulator || !simulator.running) {
        animationId = null;
        return;
    }
    
    simulator.t_sim += simulator.dt;
    updateCharts();
    
    animationId = requestAnimationFrame(animate);
}

// 차트 업데이트
function updateCharts() {
    if (!simulator || !chartManager) {
        console.error('시뮬레이터 또는 차트 매니저가 초기화되지 않았습니다.');
        return;
    }
    
    try {
        const t_days = simulator.t_sim / DAY;
        const rv = simulator.rv(simulator.t_sim);
        
        // RV 데이터 추가
        simulator.t_list.push(t_days);
        simulator.rv_list.push(rv);
        
        // 윈도우 크기 제한
        if (simulator.t_list.length > 100) {
            simulator.t_list.shift();
            simulator.rv_list.shift();
        }
        
        // 차트 업데이트
        if (chartManager.rvChart) {
            chartManager.updateRvChart(simulator.t_list, simulator.rv_list);
        }
        
        // 스펙트럼 업데이트
        const lambda_shift = simulator.lam0 * (1 + (rv / C) * DOPPLER_SCALE);
        if (chartManager.spectrumChart) {
            chartManager.updateSpectrumChart(lambda_shift, simulator.lam0);
        }
        
        // 궤도 업데이트
        chartManager.drawOrbit(simulator);
        
        // 상태 표시 업데이트
        const statusDisplay = document.getElementById('statusDisplay');
        if (statusDisplay) {
            statusDisplay.innerHTML = `RV = ${rv.toFixed(1)} m/s<br>λ = ${lambda_shift.toFixed(3)} nm (λ0=${simulator.lam0.toFixed(3)})`;
        }
    } catch (error) {
        console.error('차트 업데이트 중 오류:', error);
    }
}

// 윈도우 리사이즈 처리
window.addEventListener('resize', function() {
    if (chartManager && chartManager.orbitChart) {
        chartManager.initOrbitChart();
        updateCharts();
    }
});
