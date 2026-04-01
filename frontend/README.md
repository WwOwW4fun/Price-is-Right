# Frontend

This folder contains the React + Vite dashboard prototype for the multi-agent deal system in `react-ui/`.

The current UI uses mock data and local component state only. It includes:

- Run and stop controls for the agent system
- Status and summary cards
- A live activity panel
- A deal tracking board
- A recent alerts panel

Next step after reviewing the UI is to connect these screens to the Python backend with endpoints such as `POST /run`, `POST /stop`, `GET /status`, and `GET /deals`.
