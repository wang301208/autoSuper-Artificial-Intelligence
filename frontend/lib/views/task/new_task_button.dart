import 'package:flutter/material.dart';

class NewTaskButton extends StatelessWidget {
  final VoidCallback onPressed;
  const NewTaskButton({super.key, required this.onPressed});

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: onPressed,
      child: const Text('New Task'),
    );
  }
}
