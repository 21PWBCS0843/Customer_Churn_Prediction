document.addEventListener("DOMContentLoaded", () => {
  const navLinks = document.querySelectorAll(".sidebar-nav a")
  const contentSections = document.querySelectorAll(".content-section")
  const menuToggle = document.querySelector(".menu-toggle")
  const sidebar = document.querySelector(".sidebar")
  const mainContent = document.querySelector(".main-content")
  
  const predictionForm = document.getElementById("prediction-form")
  const predictionResult = document.getElementById("prediction-result")
  const resultPlaceholder = document.querySelector(".result-placeholder")
  const resultContent = document.querySelector(".result-content")
  const batchUploadForm = document.getElementById("batch-upload-form")
  const csvFileInput = document.getElementById("csv-file")
  const fileNameDisplay = document.getElementById("file-name")
  const batchPredictBtn = document.querySelector(".batch-predict-btn")
  const batchResults = document.getElementById("batch-results")
  const batchPlaceholder = document.querySelector(".batch-placeholder")
  const batchContent = document.querySelector(".batch-content")
  const batchLoading = document.querySelector(".batch-loading")
  const batchError = document.querySelector(".batch-error")
  const batchErrorMessage = document.getElementById("batch-error-message")
  
  // Initialize charts
  initializeCharts()
  
  // Navigation
  navLinks.forEach((link) => {
    link.addEventListener("click", function (e) {
      e.preventDefault()
      const targetId = this.getAttribute("href").substring(1)
  
      navLinks.forEach((link) => {
        link.parentElement.classList.remove("active")
      })
      this.parentElement.classList.add("active")
  
      contentSections.forEach((section) => {
        section.classList.remove("active")
        if (section.id === targetId) {
          section.classList.add("active")
        }
      })
    })
  })
  
  // Mobile menu toggle
  if (menuToggle) {
    menuToggle.addEventListener("click", () => {
      sidebar.classList.toggle("active")
      mainContent.classList.toggle("sidebar-active")
    })
  }
  
  // Individual customer prediction form
  if (predictionForm) {
    predictionForm.addEventListener("submit", (e) => {
      e.preventDefault()
  
      resultPlaceholder.style.display = "none"
      resultContent.style.display = "none"
  
      const loadingIndicator = document.createElement("div")
      loadingIndicator.className = "batch-loading"
      loadingIndicator.innerHTML = `
        <div class="spinner"></div>
        <p>Processing prediction...</p>
      `
      document.getElementById("prediction-result").appendChild(loadingIndicator)
      loadingIndicator.style.display = "flex"
  
      const formData = new FormData(predictionForm)
      const customerData = {}
      for (const [key, value] of formData.entries()) {
        if (
          key === "CreditScore" ||
          key === "Age" ||
          key === "Tenure" ||
          key === "Balance" ||
          key === "NumOfProducts" ||
          key === "EstimatedSalary"
        ) {
          customerData[key] = Number(value)
        } else if (key === "HasCrCard" || key === "IsActiveMember") {
          customerData[key] = value === "Yes" ? 1 : 0
        } else {
          customerData[key] = value
        }
      }
  
      fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(customerData),
      })
        .then((response) => {
          if (!response.ok) {
            return response.json().then((data) => {
              throw new Error(data.error || "Server error")
            })
          }
          return response.json()
        })
        .then((data) => {
          console.log("Prediction response:", data)
  
          loadingIndicator.remove()
          updatePredictionResults(data)
          resultContent.style.display = "block"
        })
        .catch((error) => {
          console.error("Error:", error)
  
          loadingIndicator.remove()
  
          const errorDiv = document.createElement("div")
          errorDiv.className = "batch-error"
          errorDiv.style.display = "flex"
          errorDiv.innerHTML = `
            <i class="fas fa-exclamation-circle"></i>
            <p>Error: ${error.message}</p>
          `
          document.getElementById("prediction-result").appendChild(errorDiv)
  
          setTimeout(() => {
            errorDiv.remove()
            if (resultPlaceholder) resultPlaceholder.style.display = "flex"
          }, 5000)
        })
    })
  }
  
  // File upload handling
  if (csvFileInput) {
    csvFileInput.addEventListener("change", function () {
      const file = this.files[0]
      if (file) {
        fileNameDisplay.textContent = file.name
        batchPredictBtn.disabled = false
      } else {
        fileNameDisplay.textContent = "No file selected"
        batchPredictBtn.disabled = true
      }
    })
  }
  
  // Batch prediction form
  if (batchUploadForm) {
    batchUploadForm.addEventListener("submit", (e) => {
      e.preventDefault()
  
      batchPlaceholder.style.display = "none"
      batchContent.style.display = "none"
      batchError.style.display = "none"
      batchLoading.style.display = "flex"
  
      const formData = new FormData(batchUploadForm)
  
      fetch("/api/batch-predict", {
        method: "POST",
        body: formData,
      })
        .then((response) => {
          if (!response.ok) {
            return response.json().then((data) => {
              throw new Error(data.error || "Server error")
            })
          }
          return response.json()
        })
        .then((data) => {
          console.log("Batch prediction response:", data)
  
          window.batchPredictionData = data
  
          batchLoading.style.display = "none"
          batchContent.style.display = "block"
  
          // Safe update with null checks
          safeUpdateElement("total-customers", data.summary?.total_customers || 0, (val) => val.toLocaleString())
          safeUpdateElement("avg-churn-risk", data.summary?.avg_churn_probability || 0, (val) => `${Math.round(val * 100)}%`)
          safeUpdateElement("batch-high-risk", data.summary?.high_risk_count || 0, (val) => val.toLocaleString())
  
          const downloadLinks = document.querySelectorAll(".download-btn[data-type='csv']")
          downloadLinks.forEach((link) => {
            link.href = `/api/download/${data.filename}`
          })
  
          const pdfDownloadBtn = document.querySelector(".download-btn[data-type='pdf']")
          if (pdfDownloadBtn) {
            pdfDownloadBtn.addEventListener("click", (e) => {
              e.preventDefault()
              generatePdfReport(data)
            })
          }
  
          updateBatchChartsFunc(data.summary)
  
          if (data.analysis) {
            updateBatchDetailedStatsFunc(data.analysis)
          }
  
          const successMessage = document.getElementById("batch-success-message")
          if (successMessage) {
            successMessage.style.display = "block"
            setTimeout(() => {
              successMessage.style.display = "none"
            }, 5000)
          }
        })
        .catch((error) => {
          console.error("Error:", error)
  
          batchLoading.style.display = "none"
          batchError.style.display = "flex"
          batchErrorMessage.textContent = error.message
        })
    })
  }
  
  // Add filter functionality
  const filterForm = document.getElementById("batch-filter-form")
  if (filterForm) {
    filterForm.addEventListener("submit", (e) => {
      e.preventDefault()
  
      if (window.batchPredictionData) {
        const filteredData = getFilteredBatchData()
      
        // Update all components with filtered data
        updateBatchResultsTable(filteredData, 1)
        updateFilteredCharts(filteredData)
        updateAllCardSections(filteredData) 
      }
    })
  
    const resetButton = filterForm.querySelector(".filter-btn.reset")
    if (resetButton) {
      resetButton.addEventListener("click", () => {
        filterForm.reset()
  
        if (window.batchPredictionData) {
          // Reset to original data
          const originalData = window.batchPredictionData.summary?.top_risk_preview || []
          updateBatchResultsTable(originalData, 1)
          updateBatchChartsFunc(window.batchPredictionData.summary)
          resetAllCardSections() // Add this new function
      
        }
      })
    }
  }
  
  fetchDashboardData()
  })
  
  // Safe update helper function
  function safeUpdateElement(elementId, value, formatter = null) {
  const element = document.getElementById(elementId)
  if (element && value !== undefined && value !== null) {
    try {
      element.textContent = formatter ? formatter(value) : value.toString()
    } catch (error) {
      console.warn(`Error updating element ${elementId}:`, error)
      element.textContent = "0"
    }
  }
  }
  
  // Safe number formatting helper
  function safeToFixed(value, decimals = 1) {
  if (value === undefined || value === null || isNaN(value)) {
    return "0"
  }
  return Number(value).toFixed(decimals)
  }
  
  // Safe percentage formatting helper
  function safePercentage(value, decimals = 1) {
  if (value === undefined || value === null || isNaN(value)) {
    return "0%"
  }
  return `${Number(value).toFixed(decimals)}%`
  }
  
  // Initialize all charts
  function initializeCharts() {
    if (typeof Chart === "undefined") {
      console.warn("Chart.js not loaded. Charts will not be initialized.")
      return
    }
    initializeSegmentChart()
    initializeRiskChart()
    initializeFactorsChart()
    initializeFinancialImpactChart() // NEW CHART INITIALIZATION
    initializeBatchRiskChart()
    initializeBatchSegmentChart()
  }
  
  // Dashboard charts
  function initializeSegmentChart() {
  const ctx = document.getElementById("segment-chart")
  if (!ctx) return
  
  window.segmentChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Young Professionals", "Established Savers", "High-Value Clients", "At-Risk Seniors"],
      datasets: [
        {
          data: [28.3, 42.7, 15.5, 13.5],
          backgroundColor: [
            "rgba(67, 97, 238, 0.7)",
            "rgba(58, 12, 163, 0.7)",
            "rgba(114, 9, 183, 0.7)",
            "rgba(247, 37, 133, 0.7)",
          ],
          borderColor: [
            "rgba(67, 97, 238, 1)",
            "rgba(58, 12, 163, 1)",
            "rgba(114, 9, 183, 1)",
            "rgba(247, 37, 133, 1)",
          ],
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          callbacks: {
            label: (context) => `${context.label}: ${context.raw}%`,
          },
        },
      },
      cutout: "70%",
    },
  })
  }
  
  function initializeRiskChart() {
  const ctx = document.getElementById("risk-chart")
  if (!ctx) return
  
  window.riskChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"],
      datasets: [
        {
          data: [45, 30, 15, 10],
          backgroundColor: [
            "rgba(76, 201, 240, 0.7)",
            "rgba(248, 150, 30, 0.7)",
            "rgba(243, 114, 44, 0.7)",
            "rgba(249, 65, 68, 0.7)",
          ],
          borderColor: ["rgba(76, 201, 240, 1)", "rgba(248, 150, 30, 1)", "rgba(243, 114, 44, 1)"],
          borderWidth: 1,
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          callbacks: {
            label: (context) => `${context.raw}% of customers`,
          },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 50,
          ticks: {
            callback: (value) => value + "%",
          },
        },
      },
    },
  })
  }
  
  function initializeFactorsChart() {
  const ctx = document.getElementById("factors-chart")
  if (!ctx) return
  
  window.factorsChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Activity Status", "Age", "Balance", "Geography", "Tenure"],
      datasets: [
        {
          data: [0.85, 0.72, 0.58, 0.45, 0.67],
          backgroundColor: "rgba(67, 97, 238, 0.7)",
          borderColor: "rgba(67, 97, 238, 1)",
          borderWidth: 1,
          borderRadius: 4,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          callbacks: {
            label: (context) => `Importance: ${context.raw.toFixed(2)}`,
          },
        },
      },
      scales: {
        x: {
          beginAtZero: true,
          max: 1,
        },
      },
    },
  })
  }
  
  function initializeFinancialImpactChart() { // NEW CHART FUNCTION
    const ctx = document.getElementById("trend-chart") // Reusing the same canvas ID
    if (!ctx) return
  
    window.financialImpactChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: ["Young Professionals", "Established Savers", "High-Value Clients", "At-Risk Seniors"],
        datasets: [
          {
            label: "Projected Savings",
            data: [0, 0, 0, 0], // Initial empty data
            backgroundColor: "rgba(76, 201, 240, 0.7)", // Blueish color
            borderColor: "rgba(76, 201, 240, 1)",
            borderWidth: 1,
            borderRadius: 4,
          },
          {
            label: "Implementation Cost",
            data: [0, 0, 0, 0], // Initial empty data
            backgroundColor: "rgba(247, 37, 133, 0.7)", // Pinkish color
            borderColor: "rgba(247, 37, 133, 1)",
            borderWidth: 1,
            borderRadius: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              boxWidth: 12,
              padding: 15,
              font: {
                size: 11,
              },
            },
          },
          tooltip: {
            callbacks: {
              label: (context) => `${context.dataset.label}: $${context.raw.toLocaleString()}`,
            },
          },
        },
        scales: {
          x: {
            stacked: false, // Not stacked, showing side-by-side bars
          },
          y: {
            beginAtZero: true,
            ticks: {
              callback: (value) => `$${value.toLocaleString()}`,
            },
          },
        },
      },
    })
  }
  
  // Batch prediction charts
  function initializeBatchRiskChart() {
  const ctx = document.getElementById("batch-risk-chart")
  if (!ctx) return
  
  window.batchRiskChart = new Chart(ctx, {
    type: "pie",
    data: {
      labels: ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"],
      datasets: [
        {
          data: [40, 30, 20, 10],
          backgroundColor: [
            "rgba(76, 201, 240, 0.7)",
            "rgba(248, 150, 30, 0.7)",
            "rgba(243, 114, 44, 0.7)",
            "rgba(249, 65, 68, 0.7)",
          ],
          borderColor: [
            "rgba(76, 201, 240, 1)",
            "rgba(248, 150, 30, 1)",
            "rgba(243, 114, 44, 1)",
            "rgba(249, 65, 68, 1)",
          ],
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            boxWidth: 12,
            padding: 15,
            font: {
              size: 11,
            },
          },
        },
        tooltip: {
          callbacks: {
            label: (context) => `${context.label}: ${safeToFixed(context.raw, 1)}%`,
          },
        },
      },
    },
  })
  }
  
  function initializeBatchSegmentChart() {
  const ctx = document.getElementById("batch-segment-chart")
  if (!ctx) return
  
  window.batchSegmentChart = new Chart(ctx, {
    type: "pie",
    data: {
      labels: ["Young Professionals", "Established Savers", "High-Value Clients", "At-Risk Seniors"],
      datasets: [
        {
          data: [30, 40, 15, 15],
          backgroundColor: [
            "rgba(67, 97, 238, 0.7)",
            "rgba(58, 12, 163, 0.7)",
            "rgba(114, 9, 183, 0.7)",
            "rgba(247, 37, 133, 0.7)",
          ],
          borderColor: [
            "rgba(67, 97, 238, 1)",
            "rgba(58, 12, 163, 1)",
            "rgba(114, 9, 183, 1)",
            "rgba(247, 37, 133, 1)",
          ],
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            boxWidth: 12,
            padding: 15,
            font: {
              size: 11,
            },
          },
        },
        tooltip: {
          callbacks: {
            label: (context) => `${context.label}: ${safeToFixed(context.raw, 1)}%`,
          },
        },
      },
    },
  })
  }
  
  // Fetch dashboard data with real values
  function fetchDashboardData() {
  fetch("/api/dashboard-data")
    .then((response) => {
      if (!response.ok) {
        throw new Error("API not available")
      }
      return response.json()
    })
    .then((data) => {
      console.log("Dashboard data:", data)
      updateDashboardWithData(data)
    })
    .catch((error) => {
      console.warn("Using fallback data:", error)
      setTimeout(() => {
        safeUpdateElement("churn-rate", 15.8, (val) => `${val}%`)
        safeUpdateElement("high-risk-count", 1245, (val) => val.toLocaleString())
      }, 1000)
    })
  }
  
  // Update dashboard with real data
  function updateDashboardWithData(data) {
  if (!data) return
  
  safeUpdateElement("churn-rate", data.churn_rate, (val) => `${safeToFixed(val, 1)}%`)
  
  const highRiskCount = (data.risk_distribution?.["High Risk"] || 0) + (data.risk_distribution?.["Very High Risk"] || 0)
  const totalCustomers = 10000
  const actualHighRiskCount = Math.round((highRiskCount / 100) * totalCustomers)
  
  safeUpdateElement("high-risk-count", actualHighRiskCount, (val) => val.toLocaleString())
  
  updateCharts(data)
  }
  
  // Update charts with real data
  function updateCharts(data) {
    if (!data) return
  
    if (window.segmentChart && data.segment_churn) {
      const segmentData = [
        data.segment_churn["Young Professionals"] || 0,
        data.segment_churn["Established Savers"] || 0,
        data.segment_churn["High-Value Clients"] || 0,
        data.segment_churn["At-Risk Seniors"] || 0,
      ]
      window.segmentChart.data.datasets[0].data = segmentData
      window.segmentChart.update()
    }
  
    if (window.riskChart && data.risk_distribution) {
      const riskData = [
        data.risk_distribution["Low Risk"] || 0,
        data.risk_distribution["Medium Risk"] || 0,
        data.risk_distribution["High Risk"] || 0,
        data.risk_distribution["Very High Risk"] || 0,
      ]
      window.riskChart.data.datasets[0].data = riskData
      window.riskChart.update()
    }
  
    if (window.factorsChart && data.top_features && data.top_features.length > 0) {
      const featureLabels = data.top_features.slice(0, 5).map((f) => f.Feature || "Unknown")
      const featureValues = data.top_features.slice(0, 5).map((f) => f.Importance || 0)
  
      window.factorsChart.data.labels = featureLabels
      window.factorsChart.data.datasets[0].data = featureValues
      window.factorsChart.update()
    }
  
    if (window.financialImpactChart && data.segment_roi) { // NEW CHART UPDATE LOGIC
      const segmentLabels = ["Young Professionals", "Established Savers", "High-Value Clients", "At-Risk Seniors"];
      const potentialRevenueLoss = segmentLabels.map(segment => data.segment_roi[segment]?.projected_savings || 0);
      const retentionCost = segmentLabels.map(segment => data.segment_roi[segment]?.implementation_cost || 0);
  
      window.financialImpactChart.data.labels = segmentLabels;
      window.financialImpactChart.data.datasets[0].data = potentialRevenueLoss;
      window.financialImpactChart.data.datasets[1].data = retentionCost;
      window.financialImpactChart.update();
    }
  }
  
  // Update prediction results with fully dynamic data
  function updatePredictionResults(data) {
  if (!data) return
  
  // Update risk gauge
  const riskGaugeFill = document.getElementById("risk-gauge-fill")
  const riskPercentage = document.getElementById("risk-percentage")
  const riskLabel = document.getElementById("risk-label")
  
  let probability, riskLevelText, segment
  
  if (data.prediction) {
    probability = data.prediction.churn_probability || 0
    riskLevelText = data.prediction.risk_level || "Low Risk"
    segment = data.prediction.customer_segment || "Unknown"
  } else {
    probability = data.churn_probability || 0
  
    if (probability >= 0.75) {
      riskLevelText = "Very High Risk"
    } else if (probability >= 0.5) {
      riskLevelText = "High Risk"
    } else if (probability >= 0.25) {
      riskLevelText = "Medium Risk"
    } else {
      riskLevelText = "Low Risk"
    }
  
    segment = data.segment || "Unknown"
  }
  
  const churnProbability = probability * 100
  if (riskGaugeFill) riskGaugeFill.style.height = `${churnProbability}%`
  if (riskPercentage) riskPercentage.textContent = `${safeToFixed(churnProbability, 1)}%`
  if (riskLabel) riskLabel.textContent = riskLevelText
  
  // Update customer profile
  safeUpdateElement("customer-segment", segment)
  safeUpdateElement("risk-level", riskLevelText)
  
  // Update similar customers with dynamic data
  const similarCustomersData = data.similar_customers || {}
  safeUpdateElement("similar-count", similarCustomersData.count || 0, (val) => val.toLocaleString())
  
  const avgChurn = similarCustomersData.avg_churn || similarCustomersData.churn_rate || 0
  safeUpdateElement("similar-churn", avgChurn, (val) => `${safeToFixed(avgChurn * 100, 1)}%`)
  
  // Update financial impact with dynamic calculations
  const financialImpactData = data.financial_impact || {}
  
  const ltv = financialImpactData.customer_ltv || financialImpactData.ltv || 0
  safeUpdateElement("customer-ltv", ltv, (val) => `$${val.toLocaleString()}`)
  
  const retentionCost = financialImpactData.retention_cost || 0
  safeUpdateElement("retention-cost", retentionCost, (val) => `$${val.toLocaleString()}`)
  
  const churnImpact = financialImpactData.churn_impact || 0
  safeUpdateElement("churn-impact", churnImpact, (val) => `$${val.toLocaleString()}`)
  
  // Update churn factors with dynamic data
  const analysisData = data.analysis || {}
  const churnFactors = analysisData.churn_factors || data.churn_factors || []
  
  const churnFactorsContainer = document.getElementById("churn-factors-container")
  if (churnFactorsContainer) {
    churnFactorsContainer.innerHTML = ""
  
    churnFactors.forEach((factor) => {
      const factorItem = document.createElement("div")
      factorItem.className = "churn-factor-item"
  
      let impactColor
      const impact = factor.impact
  
      if (typeof impact === "string") {
        if (impact === "Very High") {
          impactColor = "var(--very-high-risk)"
        } else if (impact === "High") {
          impactColor = "var(--high-risk)"
        } else if (impact === "Medium") {
          impactColor = "var(--medium-risk)"
        } else {
          impactColor = "var(--low-risk)"
        }
  
        factorItem.innerHTML = `
          <div class="factor-header">
            <div class="factor-impact" style="background-color: ${impactColor}">
              ${impact}
            </div>
            <div class="factor-name">${factor.factor || factor.name || "Unknown Factor"}</div>
          </div>
          <div class="factor-details">${factor.details || factor.description || "No details available"}</div>
        `
      } else {
        const impactValue = impact || 0
        if (impactValue >= 0.7) {
          impactColor = "var(--very-high-risk)"
        } else if (impactValue >= 0.4) {
          impactColor = "var(--high-risk)"
        } else {
          impactColor = "var(--low-risk)"
        }
  
        factorItem.innerHTML = `
          <div class="factor-header">
            <span class="factor-impact" style="background-color: ${impactColor}">${safeToFixed(impactValue * 100, 0)}%</span>
            <span class="factor-name">${factor.name || "Unknown Factor"}</span>
          </div>
          <div class="factor-details">${factor.description || "No details available"}</div>
        `
      }
  
      churnFactorsContainer.appendChild(factorItem)
    })
  }
  
  // Update retention strategies with dynamic data
  const retentionStrategies = analysisData.retention_strategies || data.retention_strategies || []
  
  const strategiesContainer = document.getElementById("strategies-container")
  if (strategiesContainer) {
    strategiesContainer.innerHTML = ""
  
    retentionStrategies.forEach((strategy) => {
      const strategyItem = document.createElement("div")
      strategyItem.className = "strategy-item"
  
      let effectivenessColor
      const effectiveness = strategy.effectiveness
  
      if (typeof effectiveness === "string") {
        if (effectiveness === "Very High") {
          effectivenessColor = "#4cc9f0"
        } else if (effectiveness === "High") {
          effectivenessColor = "#4361ee"
        } else if (effectiveness === "Medium") {
          effectivenessColor = "#7209b7"
        } else {
          effectivenessColor = "#f72585"
        }
  
        strategyItem.innerHTML = `
          <div class="strategy-header">
            <div class="strategy-name">${strategy.strategy || strategy.name || "Unknown Strategy"}</div>
            <div class="strategy-effectiveness" style="color: ${effectivenessColor}">
              ${effectiveness}
            </div>
          </div>
          <div class="strategy-details">${strategy.details || strategy.description || "No details available"}</div>
        `
      } else {
        const effectivenessValue = effectiveness || 0
        if (effectivenessValue >= 0.7) {
          effectivenessColor = "#4cc9f0"
        } else if (effectivenessValue >= 0.4) {
          effectivenessColor = "#7209b7"
        } else {
          effectivenessColor = "#f72585"
        }
  
        strategyItem.innerHTML = `
          <div class="strategy-header">
            <span class="strategy-name">${strategy.name || "Unknown Strategy"}</span>
            <span class="strategy-effectiveness" style="color: ${effectivenessColor}">${safeToFixed(effectivenessValue * 100, 0)}% Effective</span>
          </div>
          <div class="strategy-details">${strategy.description || "No details available"}</div>
        `
      }
  
      strategiesContainer.appendChild(strategyItem)
    })
  }
  
  // Update recommendations with dynamic data
  const recommendations = analysisData.recommendations || data.recommendations || []
  
  const recommendationsList = document.getElementById("recommendations-list")
  if (recommendationsList) {
    recommendationsList.innerHTML = ""
  
    recommendations.forEach((recommendation) => {
      const li = document.createElement("li")
      li.textContent = recommendation || "No recommendation available"
      recommendationsList.appendChild(li)
    })
  }
  }
  
  // Update batch charts with real data
  function updateBatchChartsFunc(summary) {
  if (!summary) return
  
  console.log("Updating batch charts with:", summary)
  
  if (window.batchRiskChart && summary.risk_distribution) {
    const riskData = [
      summary.risk_distribution["Low Risk"] || 0,
      summary.risk_distribution["Medium Risk"] || 0,
      summary.risk_distribution["High Risk"] || 0,
      summary.risk_distribution["Very High Risk"] || 0,
    ]
    window.batchRiskChart.data.datasets[0].data = riskData
    window.batchRiskChart.update()
  }
  
  if (window.batchSegmentChart && summary.segment_distribution) {
    const segmentData = [
      summary.segment_distribution["Young Professionals"] || 0,
      summary.segment_distribution["Established Savers"] || 0,
      summary.segment_distribution["High-Value Clients"] || 0,
      summary.segment_distribution["At-Risk Seniors"] || 0,
    ]
    window.batchSegmentChart.data.datasets[0].data = segmentData
    window.batchSegmentChart.update()
  }
  
  updateBatchResultsTable(summary.top_risk_preview || [])
  updateBatchStatisticsFunc(summary)
  }
  
  // Update batch statistics with real data
  function updateBatchStatisticsFunc(summary) {
  if (!summary) return
  
  if (summary.risk_distribution) {
    safeUpdateElement("low-risk-percent", summary.risk_distribution["Low Risk"], (val) => safePercentage(val, 1))
    safeUpdateElement("medium-risk-percent", summary.risk_distribution["Medium Risk"], (val) => safePercentage(val, 1))
    safeUpdateElement("high-risk-percent", summary.risk_distribution["High Risk"], (val) => safePercentage(val, 1))
    safeUpdateElement("very-high-risk-percent", summary.risk_distribution["Very High Risk"], (val) => safePercentage(val, 1))
  }
  
  if (summary.segment_distribution) {
    safeUpdateElement("yp-percent", summary.segment_distribution["Young Professionals"], (val) => safePercentage(val, 1))
    safeUpdateElement("es-percent", summary.segment_distribution["Established Savers"], (val) => safePercentage(val, 1))
    safeUpdateElement("hvc-percent", summary.segment_distribution["High-Value Clients"], (val) => safePercentage(val, 1))
    safeUpdateElement("ars-percent", summary.segment_distribution["At-Risk Seniors"], (val) => safePercentage(val, 1))
  }
  }
  
  // Update batch detailed statistics
  function updateBatchDetailedStatsFunc(analysis) {
  if (!analysis || !analysis.segment_insights) return
  
  const segmentInsights = analysis.segment_insights
  
  if (segmentInsights["Young Professionals"]) {
    const ypData = segmentInsights["Young Professionals"]
    safeUpdateElement("young-professionals-churn", ypData.avg_churn, (val) => safePercentage(val, 1))
  }
  
  if (segmentInsights["Established Savers"]) {
    const esData = segmentInsights["Established Savers"]
    safeUpdateElement("established-savers-churn", esData.avg_churn, (val) => safePercentage(val, 1))
  }
  
  if (segmentInsights["High-Value Clients"]) {
    const hvcData = segmentInsights["High-Value Clients"]
    safeUpdateElement("high-value-clients-churn", hvcData.avg_churn, (val) => safePercentage(val, 1))
  }
  
  if (segmentInsights["At-Risk Seniors"]) {
    const arsData = segmentInsights["At-Risk Seniors"]
    safeUpdateElement("at-risk-seniors-churn", arsData.avg_churn, (val) => safePercentage(val, 1))
  }
  }
  
  // Update batch results table with real data
  function updateBatchResultsTable(customers, page = 1, itemsPerPage = 5) {
  const resultsTable = document.getElementById("batch-results-table")
  if (!resultsTable || !customers || customers.length === 0) return
  
  const tbody = resultsTable.querySelector("tbody") || document.createElement("tbody")
  tbody.innerHTML = ""
  
  const startIndex = (page - 1) * itemsPerPage
  const endIndex = Math.min(startIndex + itemsPerPage, customers.length)
  const displayedCustomers = customers.slice(startIndex, endIndex)
  
  displayedCustomers.forEach((customer) => {
    let segmentClass = ""
    const segment = customer.Segment || "Unknown"
    if (segment === "Young Professionals") {
      segmentClass = "young"
    } else if (segment === "Established Savers") {
      segmentClass = "established"
    } else if (segment === "High-Value Clients") {
      segmentClass = "high-value"
    } else {
      segmentClass = "at-risk"
    }
  
    let riskClass = ""
    const riskLevel = customer.RiskLevel || "Low Risk"
    if (riskLevel === "Low Risk") {
      riskClass = "low"
    } else if (riskLevel === "Medium Risk") {
      riskClass = "medium"
    } else if (riskLevel === "High Risk") {
      riskClass = "high"
    } else {
      riskClass = "very-high"
    }
  
    const churnProb = customer.ChurnProbability || 0
    const churnProbText = typeof churnProb === "number" ? `${safeToFixed(churnProb * 100, 1)}%` : "N/A"
  
    const row = document.createElement("tr")
    row.innerHTML = `
      <td>${customer.CustomerId || "Unknown"}</td>
      <td><span class="segment-badge ${segmentClass}">${segment}</span></td>
      <td>${churnProbText}</td>
      <td><span class="risk-badge ${riskClass}">${riskLevel}</span></td>
    `
    tbody.appendChild(row)
  })
  
  if (!resultsTable.querySelector("tbody")) {
    resultsTable.appendChild(tbody)
  }
  
  updatePagination(customers.length, page, itemsPerPage)
  }
  
  // Generate PDF report with detailed insights
  function generatePdfReport(data) {
  const loadingIndicator = document.createElement("div")
  loadingIndicator.className = "batch-loading"
  loadingIndicator.style.position = "fixed"
  loadingIndicator.style.top = "0"
  loadingIndicator.style.left = "0"
  loadingIndicator.style.width = "100%"
  loadingIndicator.style.height = "100%"
  loadingIndicator.style.backgroundColor = "rgba(255, 255, 255, 0.8)"
  loadingIndicator.style.zIndex = "9999"
  loadingIndicator.innerHTML = `
    <div class="spinner"></div>
    <p>Generating detailed PDF report...</p>
  `
  document.body.appendChild(loadingIndicator)
  
  fetch("/api/batch-analysis", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ filename: data.filename }),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error("Failed to fetch analysis data")
      }
      return response.json()
    })
    .then((analysisData) => {
      const reportData = {
        summary: data.summary,
        analysis: analysisData,
        timestamp: new Date().toLocaleString(),
      }
  
      return fetch("/api/generate-pdf", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(reportData),
      })
    })
    .then((response) => {
      if (!response.ok) {
        throw new Error("Failed to generate PDF")
      }
      return response.blob()
    })
    .then((blob) => {
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.style.display = "none"
      a.href = url
      a.download = `ChurnGuard_Detailed_Report_${new Date().toISOString().slice(0, 10)}.pdf`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
  
      document.body.removeChild(loadingIndicator)
  
      const successMessage = document.getElementById("batch-success-message")
      if (successMessage) {
        successMessage.textContent = "PDF report generated successfully!"
        successMessage.style.display = "block"
        setTimeout(() => {
          successMessage.style.display = "none"
        }, 5000)
      }
    })
    .catch((error) => {
      console.error("Error generating PDF:", error)
  
      document.body.removeChild(loadingIndicator)
  
      const errorDiv = document.createElement("div")
      errorDiv.className = "batch-error"
      errorDiv.style.position = "fixed"
      errorDiv.style.top = "20px"
      errorDiv.style.left = "50%"
      errorDiv.style.transform = "translateX(-50%)"
      errorDiv.style.padding = "15px 20px"
      errorDiv.style.backgroundColor = "white"
      errorDiv.style.boxShadow = "0 4px 12px rgba(0, 0, 0, 0.15)"
      errorDiv.style.borderRadius = "8px"
      errorDiv.style.zIndex = "9999"
      errorDiv.innerHTML = `
        <i class="fas fa-exclamation-circle"></i>
        <p>Error generating PDF report: ${error.message}</p>
      `
      document.body.appendChild(errorDiv)
  
      setTimeout(() => {
        document.body.removeChild(errorDiv)
      }, 5000)
    })
  }
  
  // Update pagination
  function updatePagination(totalItems, page, itemsPerPage) {
  const paginationContainer = document.querySelector(".batch-pagination")
  if (!paginationContainer) return
  
  const totalPages = Math.ceil(totalItems / itemsPerPage)
  paginationContainer.innerHTML = ""
  
  const prevButton = document.createElement("button")
  prevButton.className = `page-btn ${page === 1 ? "disabled" : ""}`
  prevButton.innerHTML = '<i class="fas fa-chevron-left"></i>'
  if (page > 1) {
    prevButton.addEventListener("click", () => {
      if (window.batchPredictionData) {
        const filteredData = getFilteredBatchData()
        updateBatchResultsTable(filteredData, page - 1, itemsPerPage)
      }
    })
  }
  paginationContainer.appendChild(prevButton)
  
  const maxVisiblePages = 3
  const startPage = Math.max(1, page - Math.floor(maxVisiblePages / 2))
  const endPage = Math.min(totalPages, startPage + maxVisiblePages - 1)
  
  for (let i = startPage; i <= endPage; i++) {
    const pageButton = document.createElement("button")
    pageButton.className = `page-btn ${i === page ? "active" : ""}`
    pageButton.textContent = i
    pageButton.addEventListener("click", () => {
      if (window.batchPredictionData) {
        const filteredData = getFilteredBatchData()
        updateBatchResultsTable(filteredData, i, itemsPerPage)
      }
    })
    paginationContainer.appendChild(pageButton)
  }
  
  const nextButton = document.createElement("button")
  nextButton.className = `page-btn ${page === totalPages ? "disabled" : ""}`
  nextButton.innerHTML = '<i class="fas fa-chevron-right"></i>'
  if (page < totalPages) {
    nextButton.addEventListener("click", () => {
      if (window.batchPredictionData) {
        const filteredData = getFilteredBatchData()
        updateBatchResultsTable(filteredData, page + 1, itemsPerPage)
      }
    })
  }
  paginationContainer.appendChild(nextButton)
  }
  
  // Get filtered batch data
  function getFilteredBatchData() {
  if (!window.batchPredictionData || !window.batchPredictionData.summary) {
    return []
  }
  
  // Use full dataset if available, otherwise use preview
  let sourceData = window.batchPredictionData.fullDataset || window.batchPredictionData.summary.top_risk_preview || []
  let filteredData = [...sourceData]
  
  const riskFilter = document.getElementById("risk-filter")?.value
  const segmentFilter = document.getElementById("segment-filter")?.value
  const probabilityFilter = document.getElementById("probability-filter")?.value
  
  if (riskFilter && riskFilter !== "") {
    filteredData = filteredData.filter((customer) => customer.RiskLevel === riskFilter)
  }
  
  if (segmentFilter && segmentFilter !== "") {
    filteredData = filteredData.filter((customer) => customer.Segment === segmentFilter)
  }
  
  if (probabilityFilter && probabilityFilter !== "") {
    const [min, max] = probabilityFilter.split("-").map(Number)
    filteredData = filteredData.filter((customer) => {
      const probability = (customer.ChurnProbability || 0) * 100
      if (max) {
        return probability >= min && probability <= max
      } else {
        return probability >= min
      }
    })
  }
  
  return filteredData
  }
  
  // Update charts based on filtered data
  function updateFilteredCharts(filteredData) {
  if (!filteredData || filteredData.length === 0) {
    // Handle empty results - but don't update summary cards
    updateEmptyDetailedSections() // New function for non-summary cards only
    
    // Reset charts to empty state
    if (window.batchRiskChart) {
      window.batchRiskChart.data.datasets[0].data = [0, 0, 0, 0]
      window.batchRiskChart.update()
    }
    
    if (window.batchSegmentChart) {
      window.batchSegmentChart.data.datasets[0].data = [0, 0, 0, 0]
      window.batchSegmentChart.update()
    }
    
    return
  }
  
  updateDetailedCardSections(filteredData) // New function that excludes summary cards
  
  // Calculate percentages for charts
  const riskCounts = {
    "Low Risk": 0,
    "Medium Risk": 0,
    "High Risk": 0,
    "Very High Risk": 0,
  }
  
  const segmentCounts = {
    "Young Professionals": 0,
    "Established Savers": 0,
    "High-Value Clients": 0,
    "At-Risk Seniors": 0,
  }
  
  filteredData.forEach((customer) => {
    if (riskCounts.hasOwnProperty(customer.RiskLevel)) {
      riskCounts[customer.RiskLevel]++
    }
    if (segmentCounts.hasOwnProperty(customer.Segment)) {
      segmentCounts[customer.Segment]++
    }
  })
  
  const total = filteredData.length
  const riskPercentages = Object.values(riskCounts).map((count) => total > 0 ? (count / total) * 100 : 0)
  const segmentPercentages = Object.values(segmentCounts).map((count) => total > 0 ? (count / total) * 100 : 0)
  
  // Update charts
  if (window.batchRiskChart) {
    window.batchRiskChart.data.datasets[0].data = riskPercentages
    window.batchRiskChart.update()
  }
  
  if (window.batchSegmentChart) {
    window.batchSegmentChart.data.datasets[0].data = segmentPercentages
    window.batchSegmentChart.update()
  }
  }
  
  // Modified function that excludes summary cards from filtering
  function updateAllCardSections(filteredData) {
  updateDetailedCardSections(filteredData)
  }
  
  // New function that updates only detailed sections, not summary cards
  function updateDetailedCardSections(filteredData) {
  if (!filteredData || filteredData.length === 0) {
    updateEmptyDetailedSections()
    return
  }
  
  // Calculate filtered statistics
  const riskCounts = {
    "Low Risk": 0,
    "Medium Risk": 0,
    "High Risk": 0,
    "Very High Risk": 0,
  }
  
  const segmentCounts = {
    "Young Professionals": 0,
    "Established Savers": 0,
    "High-Value Clients": 0,
    "At-Risk Seniors": 0,
  }
  
  const segmentChurnData = {
    "Young Professionals": [],
    "Established Savers": [],
    "High-Value Clients": [],
    "At-Risk Seniors": []
  }
  
  filteredData.forEach((customer) => {
    // Count risk levels
    const riskLevel = customer.RiskLevel || "Low Risk"
    if (riskCounts.hasOwnProperty(riskLevel)) {
      riskCounts[riskLevel]++
    }
  
    // Count segments
    const segment = customer.Segment || "Unknown"
    if (segmentCounts.hasOwnProperty(segment)) {
      segmentCounts[segment]++
    }
    
    // Collect churn probabilities by segment
    if (segmentChurnData.hasOwnProperty(segment)) {
      segmentChurnData[segment].push(customer.ChurnProbability || 0)
    }
  })
  
  const total = filteredData.length
  
  // Calculate and update risk distribution percentages (detailed cards only)
  const riskDistribution = {}
  Object.keys(riskCounts).forEach((key) => {
    riskDistribution[key] = total > 0 ? (riskCounts[key] / total) * 100 : 0
  })
  
  // Calculate and update segment distribution percentages (detailed cards only)
  const segmentDistribution = {}
  Object.keys(segmentCounts).forEach((key) => {
    segmentDistribution[key] = total > 0 ? (segmentCounts[key] / total) * 100 : 0
  })
  
  // Update risk distribution cards (detailed breakdown)
  safeUpdateElement("low-risk-percent", riskDistribution["Low Risk"], (val) => safePercentage(val, 1))
  safeUpdateElement("medium-risk-percent", riskDistribution["Medium Risk"], (val) => safePercentage(val, 1))
  safeUpdateElement("high-risk-percent", riskDistribution["High Risk"], (val) => safePercentage(val, 1))
  safeUpdateElement("very-high-risk-percent", riskDistribution["Very High Risk"], (val) => safePercentage(val, 1))
  
  // Update segment distribution cards (detailed breakdown)
  safeUpdateElement("yp-percent", segmentDistribution["Young Professionals"], (val) => safePercentage(val, 1))
  safeUpdateElement("es-percent", segmentDistribution["Established Savers"], (val) => safePercentage(val, 1))
  safeUpdateElement("hvc-percent", segmentDistribution["High-Value Clients"], (val) => safePercentage(val, 1))
  safeUpdateElement("ars-percent", segmentDistribution["At-Risk Seniors"], (val) => safePercentage(val, 1))
  
  // Update segment-specific churn rates
  Object.keys(segmentChurnData).forEach((segment) => {
    const probabilities = segmentChurnData[segment]
    const avgChurn = probabilities.length > 0 
      ? probabilities.reduce((sum, prob) => sum + prob, 0) / probabilities.length * 100
      : 0
    
    const elementMap = {
      "Young Professionals": "young-professionals-churn",
      "Established Savers": "established-savers-churn", 
      "High-Value Clients": "high-value-clients-churn",
      "At-Risk Seniors": "at-risk-seniors-churn"
    }
    
    const elementId = elementMap[segment]
    safeUpdateElement(elementId, avgChurn, (val) => safePercentage(val, 1))
  })
  }
  
  // New function to handle empty results for detailed sections only
  function updateEmptyDetailedSections() {
  // Set all detailed percentages to 0% (but NOT summary cards)
  const detailedPercentageElements = [
    "low-risk-percent", "medium-risk-percent", "high-risk-percent", "very-high-risk-percent",
    "yp-percent", "es-percent", "hvc-percent", "ars-percent"
  ]
  
  detailedPercentageElements.forEach(elementId => {
    safeUpdateElement(elementId, 0, (val) => "0%")
  })
  
  // Set segment churn rates to 0%
  const segmentChurnElements = [
    "young-professionals-churn", "established-savers-churn", 
    "high-value-clients-churn", "at-risk-seniors-churn"
  ]
  
  segmentChurnElements.forEach(elementId => {
    safeUpdateElement(elementId, 0, (val) => "0%")
  })
  }
  
  // Handle empty results for detailed sections only (not summary cards)
  function updateEmptyCardSections() {
  updateEmptyDetailedSections()
  }
 
  function updateAdditionalCardSections(filteredData, riskDistribution, segmentDistribution) {
  const financialCards = document.querySelectorAll('.financial-metric-card')
  financialCards.forEach(card => {
    updateFinancialMetricCard(card, filteredData)
  })
  const performanceCards = document.querySelectorAll('.performance-indicator-card')
  performanceCards.forEach(card => {
    updatePerformanceIndicatorCard(card, filteredData)
  })
  }
  
  // Reset additional card sections
  function resetAdditionalCardSections() {
  // Reset any additional cards to their original state
  if (window.batchPredictionData && window.batchPredictionData.analysis) {
    const originalAnalysis = window.batchPredictionData.analysis
    
    // Reset financial cards
    const financialCards = document.querySelectorAll('.financial-metric-card')
    financialCards.forEach(card => {
      resetFinancialMetricCard(card, originalAnalysis)
    })
    
    // Reset performance cards
    const performanceCards = document.querySelectorAll('.performance-indicator-card')
    performanceCards.forEach(card => {
      resetPerformanceIndicatorCard(card, originalAnalysis)
    })
  }
  }
  
  // Helper functions for specific card types
  function updateFinancialMetricCard(card, filteredData) {
  }
  
  function resetFinancialMetricCard(card, originalAnalysis) {
  }
  
  function updatePerformanceIndicatorCard(card, filteredData) {
  }
  
  function resetPerformanceIndicatorCard(card, originalAnalysis) {
  }
  const totalItems = 0
  const currentPage = 1
  