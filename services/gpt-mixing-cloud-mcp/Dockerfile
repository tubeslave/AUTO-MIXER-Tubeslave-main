FROM node:22-bookworm-slim AS build
WORKDIR /app
COPY package*.json ./
RUN npm install --include=dev
COPY tsconfig.json ./
COPY src ./src
COPY public ./public
COPY analysis-spec.json project-state.json ./
RUN npm run build

FROM node:22-bookworm-slim AS runtime
ENV NODE_ENV=production
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY package*.json ./
RUN npm install --omit=dev && npm cache clean --force
COPY --from=build /app/dist ./dist
COPY --from=build /app/public ./public
COPY --from=build /app/analysis-spec.json ./analysis-spec.json
COPY --from=build /app/project-state.json ./project-state.json
USER node
EXPOSE 10000
CMD ["node", "dist/server.js"]
