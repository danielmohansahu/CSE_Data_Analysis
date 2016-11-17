//////////////////////////////// INITIALIZATION ////////////////////////////////
var margin = {top: 20, right: 120, bottom: 20, left: 180},
    width = 2000 - margin.right - margin.left,
    height = 480 - margin.top - margin.bottom;

var i = 0,
    root;

// Tree
var tree = d3.layout.tree()
    .size([height, width]);

// Connecting Lines
var diagonal = d3.svg.diagonal()
    .projection(function(d) { return [d.y, d.x]; });

// SVG Body
var svg = d3.select("body").append("svg")
    .attr("width", width + margin.right + margin.left)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

//////////////////////////////// UPDATE SYSTEM ////////////////////////////////

d3.json("../tree.json", function(error, flare) {
  if (error) throw error;

  root = flare;
  root.x0 = height / 2;
  root.y0 = 0;

  root.children.forEach(collapse);
  update(root);
});

//////////////////////////////// FUNCTIONS ////////////////////////////////

function update(source) {

  // Compute the new tree layout.
  var nodes = tree.nodes(root).reverse(),
      links = tree.links(nodes);

  // Normalize for fixed-depth.
  nodes.forEach(function(d) { d.y = d.depth * 100; });

  // Update the nodes…
  var node = svg.selectAll("g.node")
      .data(nodes, function(d) { return d.id || (d.id = ++i); });

  // Enter any new nodes at the parent's previous position.
  var nodeEnter = node.enter().append("g")
      .attr("class", "node")
      .on("click", click);
  nodeEnter.append("circle")

  // Add name:
  nodeEnter.append("text")
      .attr("x", function(d) { return d.children || d._children ? -10 : 10; })
      .attr("dy", ".35em")
      .attr("text-anchor", function(d) { return d.children || d._children ? "end" : "start"; })
      .text(function(d) { return d.name; })

  nodeEnter.append("title")
      .text(function(d) {return hover_text(d)});

  // Update the nodes ...
  var nodeUpdate = node
      .attr("transform", function(d) { return "translate(" + d.y + "," + d.x + ")"; });

  nodeUpdate.select("circle")
      .attr("r", 4.5)
      .style("fill", function(d) { return d._children ? "lightsteelblue" : "#fff"; });
  node.exit().remove();

  // Update the links ...
  var link = svg.selectAll("path.link")
      .data(links, function(d) { return d.target.id; })

  link.enter().insert("path", "g")
      .attr("class", "link")

  link.attr("d", diagonal);
  link.exit().remove();
}

// Toggle children on click.
function click(d) {
  if (d.children) {
    d._children = d.children;
    d.children = null;
  } else {
    d.children = d._children;
    d._children = null;
  }
  update(d);
}

function collapse(d) {
  if (d.children) {
    d._children = d.children;
    d._children.forEach(collapse);
    d.children = null;
  }
}

function hover_text(d) {
  return d.name + "\n" + "Healthy: " + d.num_negative + "\n" + "Diseased:" + d.num_positive + "\n" + "Probability of Having Heart Disease: " + d.num_positive/(d.num_positive+d.num_negative);
}