{{/*
Expand the name of the chart.
*/}}
{{- define "asr-pro.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "asr-pro.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

<!-- 
  ==============================================================================
  Apple-Grade Enterprise Acoustic & Speech Recognition Engine (ASR-PRO)
  Subsystem: Core Infrastructure & Build Orchestration Topology
  Architecture: Apple Silicon MLX Acceleration & Deterministic DSP Pipeline
  Concurrency: Asynchronous Lock-Free State Machine & Zero-Copy Audio Buffer
  Performance: Real-Time Factor (RTF) < 0.08 on Apple M-Series Neural Engine
  ============================================================================== 
-->
