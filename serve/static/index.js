(function()
{
	var canvas = document.querySelector( "#canvas" );
	var context = canvas.getContext( "2d" );
	canvas.width = 280;
	canvas.height = 280;

	var Mouse = { x: 0, y: 0 };
	var lastMouse = { x: 0, y: 0 };
	context.fillStyle="white";
	context.fillRect(0,0,canvas.width,canvas.height);
	context.color = "black";
	context.lineWidth = 10;
    context.lineJoin = context.lineCap = 'round';
	
	debug();

	canvas.addEventListener( "mousemove", function( e )
	{
		lastMouse.x = Mouse.x;
		lastMouse.y = Mouse.y;

		Mouse.x = e.pageX - this.offsetLeft;
		Mouse.y = e.pageY - this.offsetTop;

	}, false );

	canvas.addEventListener( "mousedown", function( e )
	{
		console.log("MouseDown");
		canvas.addEventListener( "mousemove", onPaint, false );


	}, false );

	canvas.addEventListener( "mouseup", function()
	{
		console.log("MouseUp");
		canvas.removeEventListener( "mousemove", onPaint, false );

	}, false );
	// Set up touch events for mobile, etc
	var Touch = { x: 0, y: 0 };
	var lastTouch = { x: 0, y: 0 };

	canvas.addEventListener("touchstart", function (e) {
  		console.log("TouchStarted");
  		canvas.addEventListener( "touchmove", onPaint, false );
	}, false);
	canvas.addEventListener("touchend", function (e) {
		canvas.removeEventListener( "touchmove", onPaint, false );
	  	console.log("TouchEnded");
	}, false);
	canvas.addEventListener("touchmove", function (e) {
		lastTouch.x=Touch.x;
		lastTouch.y=Touch.y;
		var rect = canvasDom.getBoundingClientRect();

		Touch.x=e.pageX - this.offsetLeft;
		Touch.y=e.pageY - this.offsetTop;
	}, false);

	var onPaint = function()
	{	
		context.lineWidth = context.lineWidth;
		context.lineJoin = "round";
		context.lineCap = "round";
		context.strokeStyle = context.color;
	
		context.beginPath();
		context.moveTo( lastMouse.x, lastMouse.y );
		context.lineTo( Mouse.x, Mouse.y );
		context.closePath();
		context.stroke();
	};

	function debug()
	{
		/* CLEAR BUTTON */
		var clearButton = $( "#clearButton" );
		
		clearButton.on( "click", function()
		{
			
				context.clearRect( 0, 0, 280, 280 );
				context.fillStyle="white";
				context.fillRect(0,0,canvas.width,canvas.height);
			
		});

		/* COLOR SELECTOR */

		$( "#colors" ).change(function()
		{
			var color = $( "#colors" ).val();
			context.color = color;
		});
		
		/* LINE WIDTH */
		
		$( "#lineWidth" ).change(function()
		{
			context.lineWidth = $( this ).val();
		});
	}
}());