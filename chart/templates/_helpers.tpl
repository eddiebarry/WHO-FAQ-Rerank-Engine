{{/*
Expand the name of the chart.
*/}}
{{- define "host.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "host.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "host.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels
*/}}
{{- define "host.labels" -}}
helm.sh/chart: {{ include "host.chart" . }}
{{ include "host.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Selector labels
*/}}
{{- define "host.selectorLabels" -}}
app.kubernetes.io/name: {{ include "host.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
deploymentconfig: {{ include "host.fullname" . }}
{{- end -}}

{{/*
  Create a default fully qualified mysql/postgresql name.
  We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "postgresql.fullname" -}}
{{- printf "%s-postgresql" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
  Determine the hostname to use for PostgreSQL/mySQL.
*/}}
{{- define "postgresql.hostname" -}}
{{- if .Values.postgresql.enabled -}}
{{- printf "%s-%s" "postgresql" .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s" .Values.postgresql.postgresqlServer -}}
{{- end -}}
{{- end -}}

{{/*
  Create a short redis name.
  We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "redis.name" -}}
{{- printf "redis-%s" .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
  Create a default fully qualified redis name.
  We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "redis.fullname" -}}
{{- printf "redis-%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
