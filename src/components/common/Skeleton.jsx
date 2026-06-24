import React from "react";

export const Skeleton = ({ width, height, className = "", circle = false }) => {
  const baseStyle = {
    width: width || "100%",
    height: height || "1em",
    borderRadius: circle ? "50%" : "4px",
  };

  return <div className={`skeleton-shimmer ${className}`} style={baseStyle} />;
};

export const SkeletonList = ({ count = 3, ...props }) => {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      {Array.from({ length: count }).map((_, i) => (
        <Skeleton key={i} {...props} />
      ))}
    </div>
  );
};
